# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:44:02 2020

@author: Henning
"""
import arff
import pyshark
import re
from _collections import defaultdict

# import asyncio
# import nest_asyncio
own_simulation = False
ATTACK_PROTOCOL = "udp" if own_simulation else "udp"
#ATTACKER_IDS = "209:9:9:9" if own_simulation else ["19", "12", "17", "0c", "11", "0b", "15", "16", "10", "0d"]
#ATTACKER_IDS = "209:9:9:9" if own_simulation else ["19", "18", "13"]
ATTACKER_IDS = "209:9:9:9" if own_simulation else ["0c", "0b", "09", "12", "18", "15", "13", "19", "17", "0f", "10", "11"]
BORDER_ID = "201:1:1:1" if own_simulation else "01:1:101"
ATTACK_DELAY = 480 - 1 if own_simulation else 0 # Minus a second, because to find the start of the attack, we use the first packets timestamp, which is likely not 0
ATTACK_TYPE = "sinkhole"
data = pyshark.FileCapture("SVELTE_pcaps/radiolog-1587046554377.pcap")


class flow:
    src_bytes = 0
    dst_bytes = 0
    protocol_type = ""
    flow_class = "normal"
    dio_count = 0
    dao_count = 0
    dis_count = 0

    def get_flow_as_list(self):
        return [self.src_bytes, self.dst_bytes, self.protocol, self.dio_count, self.dao_count, self.dis_count, self.flow_class]

    # This should probably be refactored at some point
    @staticmethod
    def get_flow_attributes():
        return [("src_bytes", "REAL"), ("dst_bytes", "REAL"), ("protocol_type", ["tcp", "udp", "icmpv6"]),
                ("dio_count", "REAL"), ("dao_count", "REAL"), ("dis_count", "REAL"), ("class", ["normal", "anomaly"])]


# Husk at ændre 0 og 43
transport_protocols = {
    # "0": "Hop-by-hop Option",
    "6": "tcp",
    "17": "udp",
    # "43": "Routing header for ipv6",
    "58": "icmpv6",
    "121": "smp"
}


# Packets between the 2 same hosts should be in the same flow, even if they switch up
# which host is src and which is dest. For this reason we sort the hosts (as we use their combined addresses as a key for their flow).
def sort_addresses(src_ip, dest_ip):
    sorted_list = sorted([src_ip, dest_ip], reverse=True)
    # Check if the ordering has been changed, needed later to determine src and dest bytes
    is_reversed = sorted_list[0] == dest_ip
    return is_reversed, sorted_list

#this is needed because the ips in old cooja are stupid
def get_svelte_ips(hex_ids):
    return ["::212:74" + id + ":" + id.lstrip("0") + ":" + id.lstrip("0") + id for id in hex_ids]


def get_protocol(packet):
    for key in transport_protocols:
        if transport_protocols[key] in packet:
            return transport_protocols[key]

def add_rpl_info(flow, packet):
    if "icmpv6" not in packet:
        return

    if packet.icmpv6.code == "0":
        flow.dis_count += 1
    elif packet.icmpv6.code == "1":
        flow.dio_count += 1
    elif packet.icmpv6.code == "2":
        flow.dao_count +=1
    else:
        print("unexpected icmpv6 code")


def is_packet_at_final_destination(packet):
    #The wpan address property name depends on the adress mode, so we need to check for that.
    if packet.wpan.dst_addr_mode[-1] == "3":
        #This is kinda a hacky solution, but it is necessary because the ip of border router is ::1, so cannot just match on the last 2 characters as with the other ip's
        return packet.wpan.dst64[-1] == packet.ipv6.dst[-1] and (packet.wpan.dst64[-2] == packet.ipv6.dst[-2] or
               (packet.wpan.dst64[-2] == "0" and packet.ipv6.dst[-2] == ":"))

    return False


def get_flows(raw_data):
    flow_dict = defaultdict(flow)
    is_first = True
    for packet in raw_data:
        if is_first:
            start_time = float(packet.sniff_timestamp)
            is_first = False

        #For the time being we dont want broadcast flows
        if "ipv6" in packet and "ff02" not in packet.ipv6.dst:
            is_reversed, temp_flow_identifier = sort_addresses(packet.ipv6.src, packet.ipv6.dst)
            protocol = get_protocol(packet)
            if packet.ipv6.nxt == "43" and packet.ipv6.routing_segleft != "0":
                continue

            # splits the flows into time intervals of 30 seconds
            time_since_start = float(packet.sniff_timestamp) - start_time
            time_interval = str(time_since_start - time_since_start % 30)
            flow_identifier = time_interval + ";" + temp_flow_identifier[0] + ";" + temp_flow_identifier[
                1] + ";" + protocol

            flow_dict[flow_identifier].protocol = protocol

            #To minimize duplication and stuff, we only increment bytes and packages sent, when they are actually sent to the final host
            if is_packet_at_final_destination(packet):
                flow_dict[flow_identifier].dst_bytes += int(packet.ipv6.plen) if is_reversed else 0
                flow_dict[flow_identifier].src_bytes += int(packet.ipv6.plen) if not is_reversed else 0
                add_rpl_info(flow_dict[flow_identifier], packet)

    return flow_dict


def is_anomaly(flow_key):
    if own_simulation:
        time_stamp = float(re.search("^[^;]+", flow_key).group(0))
        if time_stamp >= ATTACK_DELAY:
            if ATTACKER_IDS in flow_key and ATTACK_PROTOCOL in flow_key:
                return True

        return False

    else:
        is_blocked = False
        for ip in get_svelte_ips(ATTACKER_IDS):
            if ip in flow_key:
                is_blocked = True

        if is_blocked and ATTACK_PROTOCOL in flow_key:
            return True
        else:
            return False


def label_flows(flow_dict):
    for key in flow_dict:
        if is_anomaly(key):
            flow_dict[key].flow_class = "anomaly"


#This function returns a dictionary with the actual data carried in the packets, without any headers
def get_data_dict(raw_data):
    data_dict = defaultdict(list)

    for packet in raw_data:
        if "udp" in packet:
            data = bytearray.fromhex(packet.data.data).decode()
            data_dict[packet.ipv6.src].append(data)

    return data_dict

def get_packet_loss(packet_dict):
    received_packets = 0
    sent_packets = 0

    for key in packet_dict:
        pattern = "\(\((\\d*)\)\)"
        i = -1

        # Search for the latest packet which contains a counter
        while re.search(pattern, packet_dict[key][i]) is None:
            i -= 1

        match = re.search(pattern, packet_dict[key][i])

        if BORDER_ID in key:
            received_packets = int(match.group(1))
        else:
            sent_packets += int(match.group(1))

    return ((sent_packets - received_packets) / sent_packets) * 100

# def get_packet_loss(raw_data):
#     received_packets = 0
#     sent_packets = 0
#     for packet in raw_data:
#         if "ipv6" in packet:
#             if packet.ipv6.nxt == "43" and packet.ipv6.routing_segleft != "0":
#                 print(packet.sniff_timestamp)


def export_attacks(flow_dict, file):
    attack_list = [ATTACK_TYPE if flow_dict[key].flow_class == "anomaly" else "normal" for key in flow_dict]

    f = open(file, "w+")
    for attack in attack_list:
        f.write(attack + "\n")

    f.close()
    print("exported attacks")

def export_as_arff(flow_dict, file):
    attributes = flow.get_flow_attributes()
    data = [flow_dict[key].get_flow_as_list() for key in flow_dict]

    export_arff = {
        'relation': 'CoojaData',
        'description': 'The simulated data from the IoT environment',
        'data': data,
        'attributes': attributes
    }

    arff.dump(export_arff, open("Datasets/" + file + ".arff", "w+"))
    print("exported datasets")

    export_attacks(flow_dict, "Datasets/" + file + "_attacks")

flows = get_flows(data)
label_flows(flows)
export_as_arff(flows, "svelteSinkhole3")
print("debugging point")


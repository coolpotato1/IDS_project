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
ATTACKER_ID = "209:9:9:9"
ATTACK_DELAY = 480 - 1  # Minus a second, because to find the start of the attack, we use the first packets timestamp, which is likely not 0
data = pyshark.FileCapture("pcaps/radiolog-1585743076838.pcap")


# Probably gonna add dio, dao and dis packets later
class flow:
    src_bytes = 0
    dst_bytes = 0
    protocol_type = ""
    flow_class = "normal"

    def get_flow_as_list(self):
        return [self.src_bytes, self.dst_bytes, self.protocol, self.flow_class]

    # This should probably be refactored at some point
    @staticmethod
    def get_flow_attributes():
        return [("src_bytes", "REAL"), ("dst_bytes", "REAL"), ("protocol_type", ["tcp", "udp", "icmpv6"]),
                ("class", ["normal", "anomaly"])]


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


def get_protocol(packet):
    for key in transport_protocols:
        if transport_protocols[key] in packet:
            return transport_protocols[key]


def get_flows(raw_data):
    flow_dict = defaultdict(flow)
    is_first = True
    for packet in raw_data:
        if is_first:
            start_time = float(packet.sniff_timestamp)
            is_first = False

        if "ipv6" in packet:
            is_reversed, temp_flow_identifier = sort_addresses(packet.ipv6.src, packet.ipv6.dst)
            protocol = get_protocol(packet)
            if packet.ipv6.nxt == "43" and packet.ipv6.routing_segleft != "0":
                continue

            # splits the flows into time intervals of 30 seconds
            time_since_start = float(packet.sniff_timestamp) - start_time
            time_interval = str(time_since_start - time_since_start % 30)
            flow_identifier = time_interval + ";" + temp_flow_identifier[0] + ";" + temp_flow_identifier[1] + ";" + protocol

            flow_dict[flow_identifier].protocol = protocol
            flow_dict[flow_identifier].dst_bytes += int(packet.ipv6.plen) if is_reversed else 0
            flow_dict[flow_identifier].src_bytes += int(packet.ipv6.plen) if not is_reversed else 0

    return flow_dict


def is_attack(flow_key):
    time_stamp = float(re.search("^[^;]+", flow_key).group(0))
    if time_stamp >= ATTACK_DELAY:
        if ATTACKER_ID in flow_key and "udp" in flow_key:
            return True

    return False


def label_flows(flow_dict):
    for key in flow_dict:
        if is_attack(key):
            flow_dict[key].flow_class = "anomaly"


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

        if key == "01":
            received_packets = int(match.group(1))
        else:
            sent_packets += int(match.group(1))

    return ((sent_packets - received_packets) / sent_packets) * 100


def export_as_arff(flow_dict):
    attributes = flow.get_flow_attributes()
    data = [flow_dict[key].get_flow_as_list() for key in flow_dict]

    export_arff = {
        'relation': 'CoojaData',
        'description': 'The simulated data from the IoT environment',
        'data': data,
        'attributes': attributes
    }

    arff.dump(export_arff, open("Datasets/coojaData1.arff", "w+"))
    print("exported datasets")


flows = get_flows(data)
label_flows(flows)
export_as_arff(flows)
print("Just need something here so I can set a breaking point")

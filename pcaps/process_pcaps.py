# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:44:02 2020

@author: Henning
"""
import csv

import arff
import pyshark
import re
import numpy as np
import sys
import time
import socket
from _collections import defaultdict
from preprocessing import process_data
FLOW_LENGTH = 30
FLOW_STEP = 5

SERVER_ADDRESS = "127.0.0.1"
SERVER_PORT = 2525

CLIENT_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class flow:
    src_bytes = 0
    dst_bytes = 0
    protocol = ""
    flow_class = "normal"
    dio_count = 0
    dao_count = 0
    dis_count = 0
    usrc_bytes = 0
    udst_bytes = 0

    def get_flow_as_list(self):
        return [self.src_bytes, self.dst_bytes, self.protocol, self.dio_count, self.dao_count, self.dis_count,
                self.usrc_bytes, self.udst_bytes, self.flow_class]

    # This should probably be refactored at some point
    @staticmethod
    def get_flow_attributes():
        return [("src_bytes", "REAL"), ("dst_bytes", "REAL"), ("protocol_type", ["tcp", "udp", "icmpv6"]),
                ("dio_count", "REAL"), ("dao_count", "REAL"), ("dis_count", "REAL"), ("usrc_bytes", "REAL"), ("udst_bytes", "REAL"),
                ("class", ["normal", "anomaly"])]

    def __add__(self, other):
        return_flow = flow()
        return_flow.src_bytes = self.src_bytes + other.src_bytes
        return_flow.dst_bytes = self.dst_bytes + other.dst_bytes
        return_flow.protocol = self.protocol if self.protocol == other.protocol else None
        return_flow.dio_count = self.dio_count + other.dio_count
        return_flow.dao_count = self.dao_count + other.dao_count
        return_flow.dis_count = self.dis_count + other.dis_count
        return_flow.usrc_bytes = self.usrc_bytes + other.usrc_bytes
        return_flow.udst_bytes = self.udst_bytes + other.udst_bytes
        return return_flow


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


def csv_read(datapath):
    with open(datapath) as file:
        reader = csv.reader(file)
        data = [row for row in reader if len(row) != 0]
        return_data = []
        for row in data:
            row = row[0].split(" ")
            return_data.append([re.sub('\D', '', element) for element in row])

        return return_data


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
    elif packet.icmpv6.code == "3":
        pass
    else:
        print("unexpected icmpv6 code")


def is_packet_at_final_destination(packet):
    #The wpan address property name depends on the adress mode, so we need to check for that.
    if packet.wpan.dst_addr_mode[-1] == "3":
        #This is kinda a hacky solution, but it is necessary because the ip of border router is ::1, so cannot just match on the last 2 characters as with the other ip's
        return packet.wpan.dst64[-1] == packet.ipv6.dst[-1] and (packet.wpan.dst64[-2] == packet.ipv6.dst[-2] or
               (packet.wpan.dst64[-2] == "0" and packet.ipv6.dst[-2] == ":"))

    return False

def does_packet_fulfill_requirements(packet):
    return "ipv6" in packet and "ff02" not in packet.ipv6.dst \
           and (packet.ipv6.nxt != "43" or packet.ipv6.routing_segleft == "0")


#This uses a sliding window approach, so there is overlap between flows which is why flow_step and flow_length are not equal
#Also, best to use a flow length that is divisible for flow step (Am currently not sure if doing otherwise could cause issues
def add_packet_to_flows(flow_dict, packet, flow_length, flow_step, start_time):
    time_since_start = float(packet.sniff_timestamp) - start_time
    current_flow_step = time_since_start - time_since_start % flow_step

    # Oldest flow step goes 1 step too far back, but it evens out as np.arrange() does not include it
    # -flow_step/2 is necessary, to allow np.arange to hit 0
    oldest_flow_step_included = np.maximum(-flow_step/2, current_flow_step - flow_length)
    flows_to_add_packet = np.arange(current_flow_step, oldest_flow_step_included, -flow_step)

    # First one checks if necessary protocols are there and if it is at final destination for one protocol, second
    # one checks for other protocols if they are at final destination. Should probably be refactored to one method.
    # The reason we check for final destination, is so that we do not count a packet after every hop
    if not does_packet_fulfill_requirements(packet):
        return

    protocol = get_protocol(packet)
    is_reversed, temp_flow_identifier = sort_addresses(packet.ipv6.src, packet.ipv6.dst)

    for flow_step in flows_to_add_packet:
        flow_identifier = str(flow_step) + ";" + temp_flow_identifier[0] + ";" + temp_flow_identifier[
            1] + ";" + protocol

        flow_dict[flow_identifier].protocol = protocol

        if is_packet_at_final_destination(packet):
            if protocol != "udp":
                flow_dict[flow_identifier].dst_bytes += int(packet.ipv6.plen) if is_reversed else 0
                flow_dict[flow_identifier].src_bytes += int(packet.ipv6.plen) if not is_reversed else 0
            else:
                flow_dict[flow_identifier].udst_bytes += int(packet.ipv6.plen) if is_reversed else 0
                flow_dict[flow_identifier].usrc_bytes += int(packet.ipv6.plen) if not is_reversed else 0

            add_rpl_info(flow_dict[flow_identifier], packet)


def get_flows(raw_data):
    flow_dict = defaultdict(flow)
    is_first = True
    counter = 0
    for packet in raw_data:
        if is_first:
            start_time = float(packet.sniff_timestamp)
            is_first = False

        add_packet_to_flows(flow_dict, packet, FLOW_LENGTH, FLOW_STEP, start_time=start_time)
        counter += 1
        if counter % 1000 == 0:
            print("packets processed: ", counter)
    return flow_dict



#This function returns a dictionary with the actual data carried in the packets, without any headers
def get_data_dict(raw_data):
    data_dict = defaultdict(list)

    for packet in raw_data:
        if "udp" in packet:
            data = bytearray.fromhex(packet.data.data).decode()
            data_dict[packet.ipv6.src].append(data)

    return data_dict

# def get_packet_loss(raw_data):
#     received_packets = 0
#     sent_pack
#     for packet in raw_data:
#         if "ipv6" in packet:
#             if packet.ipv6.nxt == "43" and packet.ipv6.routing_segleft != "0":
#                 print(packet.sniff_timestamp)


def add_packet_to_live_flows(flow_dict, packet, current_time_slot, flow_step):
    protocol = get_protocol(packet)
    is_reversed, temp_flow_identifier = sort_addresses(packet.ip.src, packet.ip.dst)

    flow_identifier = str(current_time_slot) + temp_flow_identifier[0] + ";" + temp_flow_identifier[
            1] + ";" + protocol

    flow_dict[flow_identifier].protocol = protocol

    if protocol != "udp":
        flow_dict[flow_identifier].dst_bytes += int(packet.ip.plen) if is_reversed else 0
        flow_dict[flow_identifier].src_bytes += int(packet.ip.plen) if not is_reversed else 0
    else:
        flow_dict[flow_identifier].udst_bytes += int(packet.ip.len) if is_reversed else 0
        flow_dict[flow_identifier].usrc_bytes += int(packet.ip.len) if not is_reversed else 0

    add_rpl_info(flow_dict[flow_identifier], packet)


# The flow length is defined implicitly. All flow steps that are older than a given time stamp are removed,
# such that they are not included in future flow calculations
def combine_and_remove_flows(flow_dict, oldest_flow_step):
    combined_flows = defaultdict(flow)
    for key in flow_dict:
        new_key = re.sub("[^;]+;", "", key, 1)
        combined_flows[new_key] += flow_dict[key]
        if oldest_flow_step == key:
            del flow_dict[key]

    return combined_flows


def update_current_and_old_timeslot(start_time, flow_step, flow_length, packet):
    time_since_start = float(packet.sniff_timestamp) - start_time
    current_time_slot = time_since_start - time_since_start % flow_step
    oldest_time_slot = current_time_slot - (flow_length - flow_step)

    return current_time_slot, oldest_time_slot

def send_data(data):
    sent_data = data.astype(np.float32).tobytes()
    CLIENT_SOCKET.sendto(sent_data, (SERVER_ADDRESS, SERVER_PORT))
    print("message sent")

def process_and_send_flows(combined_flows, attributes, normalization_values):
    normalization_values = np.asarray(normalization_values).astype(np.float32)
    flow_list = []
    flow_names = [flow_attribute[0] for flow_attribute in flow.get_flow_attributes()]
    indexes = []
    for name in flow_names:
        indexes.append([i for i in range(len(attributes)) if attributes[i][0] == name][0])
    temp_list = ["None" if type(attribute[1]) == list else 0 for attribute in attributes]
    for key in combined_flows:
        flow_list.append(temp_list)
        new_values = combined_flows[key].get_flow_as_list()
        for i in range(len(new_values)):
            flow_list[-1][indexes[i]] = new_values[i]

    #Function returns predictions too, not needed online
    data, useless = process_data(flow_list, attributes)
    data = np.asarray(data).astype(np.float32)
    data = (data - normalization_values[0]) / normalization_values[1]
    data = np.nan_to_num(data)
    send_data(data)


# Probably update such that timestamps correspond to processed time and not sniff time, so it works with "last_sent"
def live_process_packets(flow_step, flow_length, attribute_file, normalization_file):
    flows = defaultdict(flow)
    is_first = True
    last_sent = None
    oldest_flow_step = "Not_initialized"
    capture = pyshark.LiveCapture(interface="eth1", bpf_filter="ip and udp port 80")
    attributes = arff.load(open(attribute_file))["attributes"]
    normalization_parameters = csv_read(normalization_file)


    while True:
        capture.sniff(timeout=flow_step)
        count = 0
        if len(capture) == 0:
            continue

        for packet in capture:
            if is_first:
                start_time = float(packet.sniff_timestamp)
                is_first = False
                last_sent = time.time()

            current_step, oldest_flow_step = update_current_and_old_timeslot(start_time, flow_step, flow_length, packet)
            add_packet_to_live_flows(flows, packet, current_step, flow_step)
            count +=1

            if count % 1000 == 0:
                print("packets processed ", count)

            if time.time() - last_sent > flow_step:
                combined_flows = combine_and_remove_flows(flows, oldest_flow_step)
                process_and_send_flows(combined_flows, attributes, normalization_parameters)
                last_sent = time.time()




        print("I get here")


#flows = get_flows(data)
#label_flows(flows)

#export_as_arff(flows, out_file)


live_process_packets(5, 30, "svelteSinkhole3coojaData4MitMTestKDDTest+_filtered.arff", "normalization")
print("debugging point")


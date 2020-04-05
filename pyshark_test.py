# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:44:02 2020

@author: Henning
"""
import pyshark
import re
from _collections import defaultdict

# import asyncio
# import nest_asyncio

data = pyshark.FileCapture("pcaps/radiolog-1585743076838.pcap")


class flow:
    src_bytes = 0
    dst_bytes = 0
    protocol = ""


# Husk at ændre 0 og 43
transport_protocols = {
    # "0": "Hop-by-hop Option",
    "6": "tcp",
    "17": "udp",
    # "43": "Routing header for ipv6",
    "58": "icmpv6",
    "121": "smp"
}


# Packets between the 2 same hosts should be the same, even if they switch up
# which host is src and which is dest. For this reason we sort the hosts.
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
    for packet in raw_data:
        if "ipv6" in packet:
            is_reversed, temp_flow_identifier = sort_addresses(packet.ipv6.src, packet.ipv6.dst)
            protocol = get_protocol(packet)
            if packet.ipv6.nxt == "43" and packet.ipv6.routing_segleft != "0":
                continue
            flow_identifier = temp_flow_identifier[0] + temp_flow_identifier[1] + protocol
            flow_dict[flow_identifier].protocol = protocol
            flow_dict[flow_identifier].dst_bytes += int(packet.ipv6.plen) if is_reversed else 0
            flow_dict[flow_identifier].src_bytes += int(packet.ipv6.plen) if not is_reversed else 0

    return flow_dict


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


flows = get_flows(data)
print("Just need something here so I can set a breaking point")

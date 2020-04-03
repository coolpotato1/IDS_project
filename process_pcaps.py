# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:42:39 2020

@author: Henning
"""
#import pcapkit as p
from scapy.all import *
from scapy.utils import *
from scapy.utils6 import *  # noqa: F401
from scapy.route6 import *
import re
#import dpkt
#import pyshark as p  
data = rdpcap("pcaps/radiolog-1585743076838.pcap")
#pcap_file = open("pcaps/radiolog-1585038898097.pcap", "rb")
#data = dpkt.pcap.Reader(pcap_file)

class flow:
        src_bytes = 0
        dest_bytes = 0
        protocol = ""
        
def get_node_specific_udp(packet_list):
    node_specific_udp = defaultdict(list)
    for packet in packet_list:
        if(packet.haslayer(UDP)):
            #The next line only ensures unique node ID's up till 16^2 nodes
            node_id = str(hex(packet[Dot15d4].src_addr))[-2] + str(hex(packet[Dot15d4].src_addr))[-1]
            node_specific_udp[node_id].append(str(packet[Raw].load) + "\n")
    return node_specific_udp

#Takes a packet data dictionary 
def get_counters(packet_dict):
    counters = defaultdict(list)
    for key in packet_dict:
        counters[key] = [int(re.search("\(\((\\d*)\)\)", data_str).group(1)) for data_str in packet_dict[key] if re.search("\(\((\\d*)\)\)", data_str) != None]
        
    return counters

#Packets between the 2 same hosts should be the same, even if they switch up 
#which host is src and which is dest. For this reason we sort the hosts.
def sort_addresses(src_ip, dest_ip):
    sorted_list = sorted([src_ip, dest_ip], reverse = True)
    #Check if the ordering has been changed, needed later to determine src and dest bytes
    is_reversed = sorted_list[0] == dest_ip
    return is_reversed, sorted_list
    
def get_flows(raw_data):
    flow_dict = defaultdict(flow)
    src_addresses = []
    src_addresses2 = []
    for packet in raw_data:
        if(packet.haslayer(IPv6)):
            is_reversed, temp_flow_identifier = sort_addresses(str(hex(packet[Dot15d4].src_addr)), str(hex(packet[Dot15d4].dest_addr)))
            flow_identifier = temp_flow_identifier[0] + temp_flow_identifier[1] + str(ipv6nh[packet[IPv6].nh])
            flow_dict[flow_identifier].protocol = packet[IPv6].nh
            flow_dict[flow_identifier].dest_bytes += packet[IPv6].plen if is_reversed else 0
            flow_dict[flow_identifier].src_bytes += packet[IPv6].plen if not is_reversed else 0
        if(packet.haslayer(UDP)):
            src_addresses.append(packet[IPv6].src)
            #src_addresses2.append(packet[LoWPAN].sourceAddr)
        
    return flow_dict, src_addresses, src_addresses2
        
    
def get_packet_loss(packet_dict):
     received_packets = 0
     sent_packets = 0
     
     for key in packet_dict:
         pattern = "\(\((\\d*)\)\)"
         i = -1
         
         #Search for the latest packet which contains a counter
         while(re.search(pattern, packet_dict[key][i]) == None):
             i-=1
         
         match =  re.search(pattern, packet_dict[key][i])
         
         if(key == "01"):
             received_packets = int(match.group(1))
         else:
             sent_packets += int(match.group(1))
             
     return ((sent_packets-received_packets)/sent_packets) * 100
    

#data_dict = get_node_specific_udp(data)

#print(get_packet_loss(data_dict))
     
#counters = get_counters(data_dict)
#for key in counters:
#    print(key, ": ", counters[key][-1])
flows, src_addresses, src_addresses2 = get_flows(data)



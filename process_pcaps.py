# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:42:39 2020

@author: Henning
"""
#import pcapkit as p
from scapy.all import *
from scapy.utils import *
#import dpkt
#import pyshark as p  
data = s.rdpcap("pcaps/radiolog-1585158460638.pcap")
#pcap_file = open("pcaps/radiolog-1585038898097.pcap", "rb")
#data = dpkt.pcap.Reader(pcap_file)

for packet in data:
    if(packet.haslayer(UDP)):
        print(packet.time)
        packet.show()
        break
        #print(packet[Raw].load)



# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:11:13 2020

@author: Henning
"""

import enum

class attack_types(enum.Enum):
    
    #Specific attacks
    APACHE2 = ["apache2"]
    BACK = ["back"]
    BUFFER_OVERFLOW = ["buffer_overflow"]
    FTP_WRITE = ["ftp_write"]
    GUESS_PASSWD = ["guess_passwd"]
    HTTPTUNNEL = ["httptunnel"]
    IMAP = ["imap"]
    IPSWEEP = ["ipsweep"]
    LAND = ["land"]
    LOADMODULE = ["loadmodule"]
    MAILBOMB = ["mailbomb"]
    MSCAN = ["mscan"]
    MULTIHOP = ["multihop"]
    NAMED = ["named"]
    NEPTUNE = ["neptune"]
    NMAP = ["nmap"]

    #Due to functionality in get_specific_recall, norrmal should not be included in larger attack type categories
    NORMAL = ["normal"]
    PERL = ["perl"]
    PHF = ["phf"]
    POD = ["pod"]
    PORTSWEEP = ["portsweep"]
    PROCESSTABLE = ["processtable"]
    PS = ["ps"]
    ROOTKIT = ["rootkit"]
    SAINT = ["saint"]
    SATAN = ["satan"]
    SENDMAIL = ["sendmail"]
    SMURF = ["smurf"]
    SNMPGETATTACK = ["snmpgetattack"]
    SNMPGUESS = ["snmpguess"]
    SPY = ["spy"]
    SQLATTACK = ["sqlattack"]
    TEARDROP = ["teardrop"]
    UDPSTORM = ["udpstorm"]
    WAREZCLIENT = ["warezclient"]
    WAREZMASTER = ["warezmaster"]
    WORM = ["worm"]
    XLOCK = ["xlock"]
    XSNOOP = ["xsnoop"]
    XTERM = ["xterm"]
    
    #Attack groups
    DOS = ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
                  "udpstorm", "processtable", "worm", "mailbomb"]
    
    PROBE = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]
    R2L = ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezclient", "warezmaster",
           "xsnoop", "xlock", "snmpguess", "snmpgetattack", "httptunnel", 
           "sendmail", "named", "spy"]
    
    U2R = ["buffer_overflow", "loadmodule", "rootkit", "perl", "xterm", "sqlattack", "ps"]
    
    ALL_ATTACKS_BUT_DOS = PROBE + R2L + U2R
    ALL_ATTACKS_BUT_PROBE = DOS + R2L + U2R
    ALL_ATTACKS_BUT_R2L = DOS + PROBE + U2R
    ALL_ATTACKS_BUT_U2R = DOS + PROBE + R2L
    
    KDD_ATTACKS = DOS + PROBE + R2L + U2R

    #Custom made attacks
    MITM = ["MitM"]
    MITM_NORMAL = ["MitM_normal"]
    UDP_DOS = ["UDP_DOS"]
    UDP_NORMAL = ["UDP_normal"]

    ALL_NORMALS = NORMAL + MITM_NORMAL + UDP_NORMAL

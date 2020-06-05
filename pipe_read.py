from io import BufferedReader
import sys
import socket
import numpy as np
#reader = BufferedReader(open("test", "rb"))
#bytes = reader.read()
#print(list(bytes))
HOST = '127.0.0.1'
PORT = 2525

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

while True:
    data = server_socket.recvfrom(2048)[0]
    converted_data = np.frombuffer(data, dtype=np.float32)
    print("message received")
    print("Message is: ", converted_data)

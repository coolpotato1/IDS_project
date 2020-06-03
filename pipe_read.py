from io import BufferedReader
import sys
#reader = BufferedReader(open("test", "rb"))
#bytes = reader.read()
#print(list(bytes))

while(True):
    whatever = sys.stdin.read()
    print(whatever)
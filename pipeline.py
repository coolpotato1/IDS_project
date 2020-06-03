from io import BufferedWriter, RawIOBase

writer = BufferedWriter(open("test", "wb"))
list = [3,4]
count = 0
while True:
    writer.write(bytearray(list))
    count +=1
    if count == 100:
        writer.flush()
        break

writer.close()

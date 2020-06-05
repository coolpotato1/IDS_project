from io import BufferedWriter, RawIOBase
import numpy as np
import socket


message = np.asarray([i % 2 for i in range(101)]).astype(np.float32).tobytes()

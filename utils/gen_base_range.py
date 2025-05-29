import struct
import random
import os
from collections import Counter
from statistics import mean, variance

universe = 10000   # [0, 9999]
N = 1000000        # the number of data points
output_path = f"base_intervals_0_{universe - 1}.bin"  
max_interval_length = universe 
min_interval_length = 1
random.seed(42) 
base_intervals = []
for i in range(N):
    start = random.randint(0, universe - 1)
    end = random.randint(start, universe - 1)
    while (end - start + 1)> max_interval_length or (end - start + 1) < min_interval_length:
        start = random.randint(0, universe - 1)
        end = random.randint(start, universe - 1)
    base_intervals.append((start, end))

with open(output_path, "wb") as f:
    for interval in base_intervals:
        f.write(struct.pack("ii", interval[0], interval[1]))
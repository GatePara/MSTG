import struct
import argparse
import os

def generate_base_rfann_range(n, output_path):
    base_intervals = [(i, i) for i in range(n)]

    with open(output_path, "wb") as f:
        for interval in base_intervals:
            f.write(struct.pack("ii", interval[0], interval[1]))

    print(f"Generated {n} base intervals to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base range file for RFANN.")
    parser.add_argument("output_path", type=str, help="Path to output .range file")
    parser.add_argument("num_points", type=int, help="Number of data points (intervals)")

    args = parser.parse_args()
    generate_base_rfann_range(args.num_points, args.output_path)

import struct
import random
import argparse

def generate_base_rrann_range(output_path, num_points, categories,
                               min_interval_length=1, max_interval_length=None, seed=42):
    if max_interval_length is None:
        max_interval_length = categories

    random.seed(seed)
    base_intervals = []

    for _ in range(num_points):
        while True:
            start = random.randint(0, categories - 1)
            end = random.randint(start, categories - 1)
            length = end - start + 1
            if min_interval_length <= length <= max_interval_length:
                base_intervals.append((start, end))
                break

    with open(output_path, "wb") as f:
        for interval in base_intervals:
            f.write(struct.pack("ii", interval[0], interval[1]))

    print(f"Generated {num_points} RRANN base intervals to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base range file for RRANN.")
    parser.add_argument("output_path", type=str, help="Path to output .range file")
    parser.add_argument("num_points", type=int, help="Number of data points")
    parser.add_argument("categories", type=int, help="Range category (e.g., 10000 means [0, 9999])")
    parser.add_argument("--min_len", type=int, default=1, help="Minimum interval length (inclusive)")
    parser.add_argument("--max_len", type=int, default=None, help="Maximum interval length (inclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    generate_base_rrann_range(
        args.output_path,
        args.num_points,
        args.categories,
        min_interval_length=args.min_len,
        max_interval_length=args.max_len,
        seed=args.seed
    )

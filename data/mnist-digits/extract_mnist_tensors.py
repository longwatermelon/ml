#!/usr/bin/env python
import argparse
import gzip
import random
import struct
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PIXELS = 28 * 28
CLASS_COUNT = 10
TENSOR_VALUE_CHUNK = 8192


# build command line args for the extractor
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract random MNIST subsets into Tensor serialized files"
    )
    parser.add_argument(
        "--train-examples",
        type=int,
        default=1000,
        help="number of random training examples to extract",
    )
    parser.add_argument(
        "--test-examples",
        type=int,
        default=1000,
        help="number of random test examples to extract",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for random sampling",
    )
    return parser.parse_args()


# read idx image files from the gzipped MNIST data
def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} has invalid image magic {magic}")
        if rows * cols != IMAGE_PIXELS:
            raise ValueError(f"{path} has unexpected image size {rows}x{cols}")
        data = f.read()

    expected = count * rows * cols
    if len(data) != expected:
        raise ValueError(f"{path} has {len(data)} image bytes, expected {expected}")
    return count, rows * cols, data


# read idx label files from the gzipped MNIST data
def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} has invalid label magic {magic}")
        data = f.read()

    if len(data) != count:
        raise ValueError(f"{path} has {len(data)} labels, expected {count}")
    return count, data


# choose random dataset indexes without replacement
def sample_indexes(total, requested, rng, split_name):
    if requested < 0:
        raise ValueError(f"{split_name} example count must be non-negative")
    if requested > total:
        raise ValueError(
            f"requested {requested} {split_name} examples, but only {total} exist"
        )
    return rng.sample(range(total), requested)


# count elements in a shape
def shape_numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


# write a Tensor-compatible dense double tensor
def write_tensor(path, shape, values):
    expected = shape_numel(shape)
    written = 0
    chunk = []

    with path.open("wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for dim in shape:
            f.write(struct.pack("<I", dim))
        f.write(struct.pack("<Q", expected))

        # stream data so larger requested counts do not need one giant pack
        for value in values:
            chunk.append(value)
            if len(chunk) == TENSOR_VALUE_CHUNK:
                f.write(struct.pack(f"<{len(chunk)}d", *chunk))
                written += len(chunk)
                chunk.clear()

        if chunk:
            f.write(struct.pack(f"<{len(chunk)}d", *chunk))
            written += len(chunk)

    if written != expected:
        raise ValueError(f"shape {shape} expected {expected} values, wrote {written}")


# iterate feature-major image tensor data from sampled idx records
def iter_x_values(images, indexes, pixel_count):
    for pixel in range(pixel_count):
        for image_index in indexes:
            offset = image_index * pixel_count + pixel
            yield images[offset] / 255.0


# iterate class-major one-hot label tensor data from sampled idx records
def iter_y_values(labels, indexes):
    for klass in range(CLASS_COUNT):
        for label_index in indexes:
            yield 1.0 if labels[label_index] == klass else 0.0


# extract one split into serialized x and y tensor files
def extract_split(split_name, image_name, label_name, example_count, rng):
    image_count, pixel_count, images = read_images(SCRIPT_DIR / image_name)
    label_count, labels = read_labels(SCRIPT_DIR / label_name)
    if image_count != label_count:
        raise ValueError(
            f"{split_name} image count {image_count} != label count {label_count}"
        )

    indexes = sample_indexes(image_count, example_count, rng, split_name)

    x_path = SCRIPT_DIR / f"{split_name}_X.tensor"
    y_path = SCRIPT_DIR / f"{split_name}_Y.tensor"
    write_tensor(x_path, [pixel_count, example_count], iter_x_values(images, indexes, pixel_count))
    write_tensor(y_path, [CLASS_COUNT, example_count], iter_y_values(labels, indexes))
    return x_path, y_path


# run the mnist extraction flow
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    outputs = []
    outputs.extend(
        extract_split(
            "train",
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            args.train_examples,
            rng,
        )
    )
    outputs.extend(
        extract_split(
            "test",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            args.test_examples,
            rng,
        )
    )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

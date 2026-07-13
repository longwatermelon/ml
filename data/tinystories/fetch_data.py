#!/usr/bin/env python

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    "5485261731eaac25dd8e5ebbc3839d0a9870b185/"
    "TinyStories-train.txt?download=true"
)
EXPECTED_SHA256 = "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f"
OUTPUT_PATH = Path(__file__).with_name("input.txt")
CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60


# download a URL without exposing a partial output file
def download_file(url: str, output_path: Path, expected_sha256: str) -> None:
    partial_path = output_path.with_name(f"{output_path.name}.part")
    request = Request(url, headers={"User-Agent": "tinystories-fetcher/1.0"})
    digest = hashlib.sha256()

    try:
        with urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response, partial_path.open("wb") as output_file:
            while chunk := response.read(CHUNK_SIZE):
                output_file.write(chunk)
                digest.update(chunk)

        if digest.hexdigest() != expected_sha256:
            raise OSError("downloaded file failed SHA-256 verification")

        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


# fetch the training split beside this script
def main() -> None:
    print(f"Downloading TinyStories to {OUTPUT_PATH}...")
    download_file(DATASET_URL, OUTPUT_PATH, EXPECTED_SHA256)
    print("Download complete.")


if __name__ == "__main__":
    main()

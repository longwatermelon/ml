#!/usr/bin/env python

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


DATASET_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    "b7f09a3de6d26b72b2b927415194b839b91e88b3/"
    "TinyStoriesV2-GPT4-train.txt?download=true"
)
TARGET_SIZE_BYTES = 200 * 1024 * 1024
STORY_END = b"<|endoftext|>"
EXPECTED_SUBSET_SIZE = 209_714_825
EXPECTED_SHA256 = "9b2f422a766f0f3917b7e175753cdb60a606fae2013d3d718201fdaa8575027f"
OUTPUT_PATH = Path(__file__).with_name("input.txt")
CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60


# copy a bounded prefix and trim it to the last complete story
def copy_story_prefix(source, output, target_size: int, story_end: bytes) -> int:
    downloaded = 0
    last_story_end = 0
    tail = b""

    while downloaded < target_size:
        chunk = source.read(min(CHUNK_SIZE, target_size - downloaded))
        if not chunk:
            raise OSError("dataset ended before the requested subset size")

        output.write(chunk)
        searchable = tail + chunk
        searchable_start = downloaded - len(tail)
        search_pos = 0
        while (marker_pos := searchable.find(story_end, search_pos)) != -1:
            last_story_end = searchable_start + marker_pos + len(story_end)
            search_pos = marker_pos + len(story_end)

        tail = searchable[-(len(story_end) - 1):]
        downloaded += len(chunk)

    if last_story_end == 0:
        raise OSError("downloaded subset contains no complete stories")

    output.truncate(last_story_end)
    return last_story_end


# hash a file without loading it all into memory
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


# download a verified subset without exposing a partial output file
def download_subset(url: str, output_path: Path) -> None:
    partial_path = output_path.with_name(f"{output_path.name}.part")
    request = Request(url, headers={"User-Agent": "tinystories-fetcher/1.0"})

    try:
        with urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response, partial_path.open("wb") as output_file:
            subset_size = copy_story_prefix(
                response, output_file, TARGET_SIZE_BYTES, STORY_END
            )

        if subset_size != EXPECTED_SUBSET_SIZE:
            raise OSError("downloaded subset ended at an unexpected story boundary")
        if file_sha256(partial_path) != EXPECTED_SHA256:
            raise OSError("downloaded file failed SHA-256 verification")

        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


# fetch the training split beside this script
def main() -> None:
    print(f"Downloading a 200 MiB TinyStories V2 GPT-4 subset to {OUTPUT_PATH}...")
    download_subset(DATASET_URL, OUTPUT_PATH)
    print("Download complete.")


if __name__ == "__main__":
    main()

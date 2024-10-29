import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils.video import get_h_w_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("crop_info", type=str)
    parser.add_argument("video_folder", type=str)
    parser.add_argument("--min_size", type=int, default=768)
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    crop_info_file = Path(args.crop_info)
    crop_info = crop_info_file.open("r")
    filtered_crop_path = crop_info_file.with_stem(f"filtered-{str(args.min_size)}-{crop_info_file.stem}")
    filtered_crop = filtered_crop_path.open("w")

    cached_video_info = {}
    filtered_video_ids = defaultdict(set)
    lines = crop_info.readlines()
    for line in tqdm(lines, total=len(lines)):
        line = line.strip()

        video_name_with_split, H, W, S, E, L, T, R, B = line.split(',')
        H, W, S, E, L, T, R, B = int(H), int(W), int(S), int(E), int(L), int(T), int(R), int(B)

        video_name = ("_").join(video_name_with_split.split("_")[:-1])
        video_split = video_name_with_split.split("_")[-1]

        input_filepath = Path(video_folder / f"{video_name}.mp4")
        if not input_filepath.exists():
            tqdm.write(f"{input_filepath} not found")
            continue

        if input_filepath not in cached_video_info:
            cached_video_info[input_filepath] = get_h_w_fps(input_filepath)
        h, w, fps = cached_video_info[input_filepath]
        if E - S < fps:
            continue

        t = int(T / H * h)
        b = int(B / H * h)
        l = int(L / W * w)
        r = int(R / W * w)

        if (b - t) < args.min_size or (r - l) < args.min_size:
            continue

        filtered_video_ids[video_name].add(video_split)
        filtered_crop.write(line + "\n")
        tqdm.write("write " + line)

    crop_info.close()
    filtered_crop.close()
    print(f"save to {filtered_crop_path}")
    filtered_crop_info_path = crop_info_file.parent / f"filtered-{str(args.min_size)}-splits.txt"
    with filtered_crop_info_path.open("w") as f:
        for k, v in filtered_video_ids.items():
            v = list(sorted(v))
            f.write(f"{k}," + ",".join(v) + "\n")
    print(f"save to {filtered_crop_info_path}")



if __name__ == "__main__":
    main()

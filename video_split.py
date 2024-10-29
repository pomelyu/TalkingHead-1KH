import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split_info", type=str, help="splits file")
    parser.add_argument("video_folder", type=str)
    parser.add_argument("save_folder", type=str)
    args = parser.parse_args()

    video_folder = Path(args.video_folder)
    save_folder = Path(args.save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    split_info = Path(args.split_info).open("r")
    while True:
        line = split_info.readline().strip()
        if len(line) == 0:
            break

        splits = line.split(",")
        video_name = splits[0]
        cmd = [
            "ffmpeg",
            "-i", f"{video_folder / video_name}.mp4",
            "-c", "copy",
            "-map", "0",
            "-segment_time", "00:01:00",
            "-f", "segment",
            f"{save_folder / video_name}_%04d.mp4"
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

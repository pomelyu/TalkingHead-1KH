from pathlib import Path

SPLIT = 200

video_ids = Path("data_list/train_video_ids.txt").open("r")
video_tubes = Path("data_list/train_video_tubes.txt").open("r")

counter = 0
sub_video_ids = None
sub_video_tubes = None
video_line = None
while True:
    line = video_ids.readline()
    if len(line) == 0:
        break

    if counter % SPLIT == 0:
        i = counter // SPLIT
        if sub_video_ids != None:
            sub_video_ids.close()
            sub_video_tubes.close()

        print(f"create split {i:0>3d}")
        folder = Path(f"data_list/train/train_split_{i:0>3d}")
        folder.mkdir(parents=True, exist_ok=True)
        sub_video_ids = (folder / "train_video_ids.txt").open("w+")
        sub_video_tubes = (folder / "train_video_tubes.txt").open("w+")

    sub_video_ids.write(line)
    video_name = line.rstrip()

    if video_line != None:
        sub_video_tubes.write(video_line)

    while video_line := video_tubes.readline():
        if len(video_line) == 0:
            sub_video_tubes.close()
            break

        if video_name in video_line:
            sub_video_tubes.write(video_line)
        else:
            break

    counter += 1

video_ids.close()

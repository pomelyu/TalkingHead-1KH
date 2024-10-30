# Download the videos.
python videos_download.py \
    --input_list data_list/train/train_split_003/train_video_ids.txt \
    --output_dir data_list/train/train_split_003/raw_videos

# Select clips based on image resolution.
python filter_datalist.py \
    data_list/train/train_split_003/train_video_tubes.txt \
    data_list/train/train_split_003/raw_videos \
    --min_size=512 \

# Split the videos into 1-min chunks.
python video_split.py .\
    data_list/train/train_split_003/filtered-512-splits.txt \
    data_list/train/train_split_003/raw_videos \
    data_list/train/train_split_003/1min_clips \

# Extract the talking head clips.
python videos_crop.py \
    --input_dir data_list/train/train_split_003/1min_clips/ \
    --output_dir data_list/train/train_split_003/cropped_clips \
    --clip_info_file data_list/train/train_split_003/filtered-512-train_video_tubes.txt

python filter_datalist.py \
    data_list/train/train_split_003/train_video_tubes.txt \
    data_list/train/train_split_003/raw_videos \
    --min_size=512 \

python video_split.py .\
    data_list/train/train_split_003/filtered-512-splits.txt \
    data_list/train/train_split_003/raw_videos \
    data_list/train/train_split_003/1min_clips \

dataset=$1
save_folder=data_list/train/train_split_$2

# # Download the videos.
# python videos_download.py \
#     --input_list ${save_folder}/${dataset}_video_ids.txt \
#     --output_dir ${save_folder}/raw_videos

# # Split the videos into 1-min chunks.
# ./videos_split.sh ${save_folder}/raw_videos ${save_folder}/1min_clips

# Extract the talking head clips.
python videos_crop.py \
    --input_dir ${save_folder}/1min_clips/ \
    --output_dir ${save_folder}/cropped_clips \
    --clip_info_file ${save_folder}/${dataset}_video_tubes.txt
import ffmpeg


def get_h_w_fps(filepath):
    probe = ffmpeg.probe(filepath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    height = int(video_stream['height'])
    width = int(video_stream['width'])
    fps = eval(video_stream["r_frame_rate"])
    return height, width, fps

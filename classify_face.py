import argparse
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from tqdm import tqdm

from utils.date import get_datetime
from utils.io import load_video_start_and_end, write_formatted_json
from utils.os import files_in_folder, mkdir, non_interruptable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafolder", type=str, help="path to image folder")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument("--min_frames", type=int, default=30, help="discard videos with fewer frames")
    parser.add_argument("--min_face_ratio", type=float, default=0.3, help="minimum face size compare to image size")
    parser.add_argument("--similarity_thres", type=float, default=0.6, help="similarity threshold for face recognition")
    parser.add_argument("--id_offset", type=int, default=0, help="offset id in database")
    args = parser.parse_args()

    datetime = get_datetime()
    output = mkdir(args.output)
    trash_folder = mkdir(output / "trash")
    debug_folder = mkdir(output / "debug")
    database_path = output / "id_database.npy"

    write_formatted_json(output / f"cmd_{datetime}.json", vars(args))

    datafolder = Path(args.datafolder)
    if database_path.exists():
        shutil.copy(database_path, database_path.with_name(f"{database_path.stem}_{datetime}{database_path.suffix}"))
        id_database = np.load(database_path, allow_pickle=True).item()
    else:
        id_database = {}

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    meta_path = output / "meta.txt"
    meta_file = Path(meta_path).open("a+")

    video_files = files_in_folder(datafolder, ext=".mp4")
    for video_path in tqdm(video_files, total=len(video_files)):
        try:
            start, mid, end, num_frames = load_video_start_and_end(video_path)
        except RuntimeError as err:
            tqdm.write(str(err))
            shutil.move(video_path, trash_folder)
            continue

        debug_images = [start, mid, end]
        debug_path = str(debug_folder / f"{Path(video_path).stem}.jpg")
        if num_frames < args.min_frames:
            discard_video(video_path, trash_folder, debug_path, debug_images,
                          reason=f"Too few frames: {num_frames} < {args.min_frames}")
            continue

        # Validate video
        # 1. Face exists
        face_start, reason_start = get_largest_face(app, start, min_face_ratio=args.min_face_ratio)
        face_mid, reason_mid = get_largest_face(app, mid, min_face_ratio=args.min_face_ratio)
        face_end, reason_end = get_largest_face(app, end, min_face_ratio=args.min_face_ratio)

        if face_start is None:
            discard_video(video_path, trash_folder, debug_path, debug_images, reason=f"start frame {reason_start}")
            continue

        if face_mid is None:
            discard_video(video_path, trash_folder, debug_path, debug_images, reason=f"mid frame {reason_mid}")
            continue

        if face_end is None:
            discard_video(video_path, trash_folder, debug_path, debug_images, reason=f"end frame {reason_end}")
            continue

        vec_start = face_start.normed_embedding
        vec_mid = face_mid.normed_embedding
        vec_end = face_end.normed_embedding

        # 2. Faces in start and end frame are the same
        similarity = np.dot(vec_start, vec_mid)
        if similarity < args.similarity_thres:
            discard_video(video_path, trash_folder, debug_path, debug_images,
                          reason=f"faces(start, mid) are not consistent: score {similarity:0.2f}")
            continue

        similarity = np.dot(vec_mid, vec_end)
        if similarity < args.similarity_thres:
            discard_video(video_path, trash_folder, debug_path, debug_images,
                          reason=f"faces(mid, end) are not consistent: score {similarity:0.2f}")
            continue

        vec_video = normalize_vec(vec_start + vec_mid + vec_end)

        # Classify face
        is_classified = False
        for id, vec in id_database.items():
            similarity = np.dot(vec_video, vec)
            if similarity >= args.similarity_thres:
                with non_interruptable():
                    shutil.move(video_path, output / id)
                    id_database[id] = normalize_vec(vec * 0.9 + vec_video * 0.1)
                    np.save(database_path, id_database)
                    meta_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        Path(video_path).stem,
                        id,
                        *face_start.bbox.astype(np.int32),
                        *face_mid.bbox.astype(np.int32),
                        *face_end.bbox.astype(np.int32),
                    ))
                is_classified = True
                break

        if is_classified:
            continue

        new_id = f"id_{len(id_database) + args.id_offset:0>6d}"
        new_folder = mkdir(output / new_id)
        with non_interruptable():
            shutil.move(video_path, new_folder)
            id_database[new_id] = vec_video
            np.save(database_path, id_database)
            meta_file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                Path(video_path).stem,
                new_id,
                *face_start.bbox.astype(np.int32),
                *face_mid.bbox.astype(np.int32),
                *face_end.bbox.astype(np.int32),
            ))

    meta_file.close()

def discard_video(video_path, trash_folder, debug_path, debug_images, reason):
    write_debug_image(debug_path, debug_images, reason)
    tqdm.write(f"{reason}: {video_path}")
    shutil.move(video_path, trash_folder)

def write_debug_image(path, debug_images, reason: str = None):
    debug_images = np.concatenate(debug_images, 1)
    H = debug_images.shape[0]
    if reason is not None:
        cv2.putText(debug_images, reason, (0, H), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    cv2.imwrite(path, debug_images)

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

def get_largest_face(app: FaceAnalysis, image: np.ndarray, min_face_ratio: float = 0.5) -> Face:
    faces: List[Face] = app.get(image)
    if len(faces) == 0:
        return None, "Face not found"

    # Face(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
    largest = None
    largest_size = 0
    for face in faces:
        x0, y0, x1, y1 = face.bbox
        size = max(x1 - x0, y1 - y0) 
        if size > largest_size:
            largest_size = size
            largest = face

    H, W = image.shape[:2]
    x0, y0, x1, y1 = largest.bbox.astype(np.int32)
    cv2.rectangle(image, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=1)
    if largest_size < max(H, W) * min_face_ratio:
        return None, f"face too small, {int(largest_size)} vs {max(H, W)}"

    return largest, "success"

if __name__ == "__main__":
    main()

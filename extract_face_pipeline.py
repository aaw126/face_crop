import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import tempfile
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import shutil
from multiprocessing import Process, Semaphore
import argparse

max_processes = 10 
semaphore = Semaphore(max_processes)

def extract_image_region(img, x, y, x2, y2):
    h, w = img.shape[:2]
    channels = 3 if len(img.shape) == 3 else 1 
    
    result_h = max(y2 - y, 0)
    result_w = max(x2 - x, 0)
    canvas = np.zeros((result_h, result_w, channels), dtype=np.uint8) if channels > 1 \
        else np.zeros((result_h, result_w), dtype=np.uint8)
    
    x_start = max(x, 0)
    x_end = min(x2, w)
    y_start = max(y, 0)
    y_end = min(y2, h)
    
    canvas_x_start = x_start - x
    canvas_x_end = canvas_x_start + (x_end - x_start)
    canvas_y_start = y_start - y
    canvas_y_end = canvas_y_start + (y_end - y_start)

    if (x_end > x_start) and (y_end > y_start):
        if channels == 1:
            valid_region = img[y_start:y_end, x_start:x_end]
            canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = valid_region
        else:
            valid_region = img[y_start:y_end, x_start:x_end, :]
            canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end, :] = valid_region

    return canvas


def interpolate_missing_frames(center_points, max_missing=5):
    valid_indices = [i for i, p in enumerate(center_points) if p is not None]
    if not valid_indices:
        return None

    x_vals = [center_points[i][0] for i in valid_indices]
    y_vals = [center_points[i][1] for i in valid_indices]

    x_interp = interp1d(valid_indices, x_vals, kind='linear', fill_value="extrapolate")
    y_interp = interp1d(valid_indices, y_vals, kind='linear', fill_value="extrapolate")

    return [(int(x_interp(i)), int(y_interp(i))) if center_points[i] is None else center_points[i]
            for i in range(len(center_points))]


def get_frame_roi(face_landmarks_list, image_width, image_height):
    all_x, all_y = [], []

    for face_landmarks in face_landmarks_list:
        x_coords = [landmark.x * image_width for landmark in face_landmarks]
        y_coords = [landmark.y * image_height for landmark in face_landmarks]
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    if not all_x or not all_y:
        return None, None, None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    face_width = x_max - x_min
    face_height = y_max - y_min
    side_length = max(face_width, face_height) * 1.5 

    return int(center_x), int(center_y), int(side_length)

def crop_face_from_video(video_path, output_file, detector):
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_video_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    max_side_length = 0
    center_points = []
    all_detection_result = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(image)
        all_detection_result.append(detection_result)
        if detection_result.face_landmarks:
            cx, cy, side_length = get_frame_roi(detection_result.face_landmarks, frame_width, frame_height)
            max_side_length = max(max_side_length, side_length)
            center_points.append((cx, cy))
        else:
            center_points.append(None)

    cap.release()

    if all(p is None for p in center_points):
        print(f"所有帧都未检测到人脸，舍弃视频: {video_path}")
        return
        
    interpolated_points = interpolate_missing_frames(center_points)

    if interpolated_points is None:
        print(f"视频无效: {video_path}")
        return
        # split_video(video_path, detect_scenes(video_path), os.path.dirname(output_path))
        # return

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (max_side_length, max_side_length))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        center_x, center_y = interpolated_points[frame_idx]
        frame_idx += 1
        cropped_frame = extract_image_region(frame, center_x - max_side_length // 2, center_y - max_side_length // 2,
                                             center_x + max_side_length // 2, center_y + max_side_length // 2)
        # x_min = max(0, center_x - max_side_length // 2)
        # y_min = max(0, center_y - max_side_length // 2)
        # x_max = min(frame_width, x_min + max_side_length)
        # y_max = min(frame_height, y_min + max_side_length)

        # cropped_frame = frame[y_min:y_max, x_min:x_max]

        out.write(cropped_frame)

    cap.release()
    if out:
        out.release()

    original_clip = VideoFileClip(video_path)
    processed_clip = VideoFileClip(temp_video_path).set_audio(original_clip.audio)
    processed_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    os.remove(temp_video_path)
    # shutil.rmtree(temp_video_path)


def detect_scenes(video_path, threshold=30):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def process_single_video(input_video_path, output_dir, detector):
    # with semaphore:
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]  # 获取原视频名
    for i in range(10):
        if os.path.exists(os.path.join(output_dir, f"{video_name}_scene_{i}.mp4")):
            print(f"{video_name} has been processed")
            return
    scene_list = detect_scenes(input_video_path)

    video_clip = VideoFileClip(input_video_path)
    video_duration = video_clip.duration 
    if scene_list == []:
        scene_list.append((0, video_duration))
    for i, (start_time, end_time) in enumerate(scene_list):
        start_time = min(start_time, video_duration)
        end_time = min(end_time, video_duration)

        if start_time >= end_time:
            print(f"跳过无效时间范围: {start_time}-{end_time} 秒")
            continue
            
        temp_clip_path = os.path.join(output_dir, f"{video_name}_scene_{i}_temp.mp4")
        clip = video_clip.subclip(start_time, end_time)
        clip.write_videofile(temp_clip_path, codec='libx264', audio_codec='aac')
        crop_face_from_video(temp_clip_path, os.path.join(output_dir, f"{video_name}_scene_{i}.mp4"), detector)
        os.remove(temp_clip_path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    args=parser.parse_args()
    input_folder=args.input_dir
    output_folder=args.output_dir
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1            
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    processes = []
    files = os.listdir(input_folder)
    for file in tqdm(files):
        # p = Process(target=process_single_video, args=(os.path.join(input_folder, file), output_folder, detector))
        # p.start()
        # processes.append(p)

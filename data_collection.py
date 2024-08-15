"""
Objective of given inputed file of video footage have 
code break down each frame into its key points and then
have these then returned into usable numpy arrays for h5 file
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize MediaPipe and YOLO models
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
yolo_model = YOLO('yolov8l.pt')

# Detect keypoints using MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                   
    results = model.process(image)                  
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return results

# Detect objects using YOLO
def yolo_detection(image):
    results = yolo_model(image)
    return results

# Extract keypoints from MediaPipe results
def extract_numpy_keypoints(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    left_leg = [
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z]
    ] if results.pose_landmarks else np.zeros((4, 3))

    right_leg = [
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z],
        [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z]
    ] if results.pose_landmarks else np.zeros((4, 3))

    left_leg = np.array(left_leg).flatten()
    right_leg = np.array(right_leg).flatten()

    # Concatenate all keypoints into a single numpy array
    keypoints = np.concatenate([pose, left_leg, right_leg])

    return keypoints

# Extract ball keypoints from YOLO results
def extract_ball_keypoints(results):
    ball_keypoints = []
    for result in results:
        # Ensure the result contains bounding boxes and class IDs
        if 'boxes' in result and 'class_ids' in result:
            boxes = result['boxes']
            class_ids = result['class_ids']
            for box, class_id in zip(boxes, class_ids):
                if class_id == 32:  
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    ball_keypoints.append([x_center, y_center])
    
    return np.array(ball_keypoints).flatten() if ball_keypoints else np.zeros(2)

# Data path and setup
DATA_PATH = os.path.join('SHOOTING_DATA')
actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])
no_videos = 15
frame_per_video = 90

for action in actions:
    for video in range(no_videos):
        os.makedirs(os.path.join(DATA_PATH, action, str(video)), exist_ok=True)

# Process video
def process_video(video_path, action, video_idx):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)
            yolo_results = yolo_detection(frame)

            # Extract keypoints
            keypoints = extract_numpy_keypoints(results)
            ball_keypoints = extract_ball_keypoints(yolo_results)
            combined_keypoints = np.concatenate([keypoints, ball_keypoints])

            frames.append(combined_keypoints)

        cap.release()

        # Standardize the number of frames
        frames = standardize_frames(frames, frame_per_video)

        for frame_idx, frame in enumerate(frames):
            # Save keypoints
            npy_path = os.path.join(DATA_PATH, action, str(video_idx), f"{frame_idx}.npy")
            np.save(npy_path, frame)

def standardize_frames(frames, target_frame_count):
    if len(frames) == target_frame_count:
        return frames
    elif len(frames) < target_frame_count:
        # Pad with the last frame
        while len(frames) < target_frame_count:
            frames.append(frames[-1])
    else:
        # Truncate to target_frame_count
        frames = frames[:target_frame_count]
    return frames

# Main loop to process videos
video_directory = r'C:\Users\samue\OneDrive\Documents\Programming\Projects\footy_ai\raw_video'

for action in actions:
    action_dir = os.path.join(video_directory, action)
    if not os.path.exists(action_dir):
        print(f"Directory {action_dir} does not exist. Skipping.")
        continue
    for video_idx, video_file in enumerate(os.listdir(action_dir)):
        if video_file.endswith(('.mp4', '.avi', '.MOV')) and video_idx < no_videos:
            video_path = os.path.join(action_dir, video_file)
            print(f"Processing video: {video_file} in action: {action}")
            process_video(video_path, action, video_idx)

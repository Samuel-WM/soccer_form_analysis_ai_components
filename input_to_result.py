"""
This file can take any video path provided and as long as 
the user has access to the trained model they can process the
video and recieve a prediction. This iteration will take longer
as the YOLOv8 object detection has to process each frame
which slows down the run time of the program signifiantly
"""


import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from ultralytics import YOLO


actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])

#defining out the specific video we want to process
video_path = r'C:\Users\samue\OneDrive\Documents\Programming\Projects\footy_ai\ai_components\raw_video\ok_shot\IMG_1785.MOV'


#defining the models we need
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
yolo_model = YOLO('yolov8l.pt')

#either pad or truncate video to be 90 frames
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



"""
Goal of next section is to be able to create functions
that can be used to generate numpy arrays for the given video
that can be used as inputs to the CNN
"""

#detect keypoints
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                   
    results = model.process(image)                  
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return results

#function to turn keypints in each frame into numpy arrays
def extract_form_keypoints(results):
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


#detct the soccer ball
def yolo_detection(image):
    results = yolo_model(image)
    return results


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




#function will bring together the rest of code for cohesive process
def process_video(video_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)
            yolo = yolo_detection(frame)

            keypoints = extract_form_keypoints(results)
            ball_keypoints = extract_ball_keypoints(yolo)

            combined_keypoints = np.concatenate([keypoints, ball_keypoints])

            frames.append(combined_keypoints)

        cap.release()

        frames = standardize_frames(frames, 90)

        return np.array(frames)
        
"""
Creating an inctance of the ai model so that we can process and predict
the result of the users imputted video
"""

#defining what model to use

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.dot(x, self.W) + self.b
        e = tf.keras.backend.tanh(e)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
cnn_model = load_model('shooting_form_v2.h5', custom_objects={'AttentionLayer': AttentionLayer})

numpy_array = process_video(video_path)

if numpy_array.shape != (1, 90, 158):
    numpy_array = numpy_array.reshape(1, 90, 158)

prediction = cnn_model.predict(numpy_array)

predicted_index = np.argmax(prediction)

predicted_label = actions[predicted_index]

#Code to write out result to seperate file
with open('footy-ai-analysis-result.txt', 'w') as file:
    file.write(f"The result of your video was {predicted_label}")

#print result to terminal
print(f"The result of your video was {predicted_label}")
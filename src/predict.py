import cv2
import numpy as np
import pandas as pd
from collections import deque, Counter
import mediapipe as mp
import logging
import time

logger = logging.getLogger(__name__)


def predict(model, le, detector, frame, delay=0):
    if delay > 0:
        time.sleep(delay)

    try:
        width = frame.shape[1]
        height = frame.shape[0]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)
        landmarks = results.multi_hand_landmarks

        if not landmarks:
            logger.warning("No hand detected.")
            return None

        hand_landmarks = landmarks[0].landmark
        logger.info(f"Total landmarks detected: {len(hand_landmarks)}")

        for i, lm in enumerate(hand_landmarks):
            logger.debug(f"Landmark {i}: x={lm.x}, y={lm.y}, z={lm.z}")


        data = {}
        for i in range(21):
            data[f'x{i+1}'] = [hand_landmarks[i].x * width]
            data[f'y{i+1}'] = [hand_landmarks[i].y * height]

        df = pd.DataFrame(data)
        logger.info(f"Raw DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Raw DataFrame preview:\n{df.head()}")

        df_processed = preprocess_hand_landmarks(df, img_width=width, img_height=height)
        logger.info(f"Processed DataFrame columns: {df_processed.columns.tolist()}")
        logger.debug(f"Processed DataFrame preview:\n{df_processed.head()}")

        features = []
        for i in range(1, 22):
            features.append(df_processed[f'x{i}'].values[0])
            features.append(df_processed[f'y{i}'].values[0])

        logger.debug(f"Feature vector: {features}")
        
        prediction = model.predict([features])[0]
        label = le.inverse_transform([prediction])[0]

        logger.info(f"Predicted gesture: {label}")
        return label

    except Exception as e:
        logger.error(f"Gesture prediction failed: {str(e)}")
        raise ValueError(f"Prediction error: {str(e)}")

def predict_from_landmarks(model, le, landmarks):
    try:
        logger.debug(f"Received landmarks: {[l.dict() for l in landmarks]}")
        
        data = {
            **{f'x{i+1}': [landmarks[i].x] for i in range(21)},
            **{f'y{i+1}': [landmarks[i].y] for i in range(21)}
        }


        df = pd.DataFrame(data)

        logger.info(f"Raw landmark DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Raw landmark DataFrame preview:\n{df.head()}")

        df_processed = preprocess_hand_landmarks(df)

        features = []
        for i in range(1, 22):
            features.append(df_processed[f'x{i}'].values[0])
            features.append(df_processed[f'y{i}'].values[0])

        prediction = model.predict([features])[0]
        label = le.inverse_transform([prediction])[0]

        logger.info(f"Predicted gesture from landmarks: {label}")
        return label

    except Exception as e:
        logger.error(f"Landmark prediction failed: {str(e)}")
        raise ValueError(f"Prediction error: {str(e)}")


def preprocess_hand_landmarks(df, img_width=1920, img_height=1080, num_points=21, palm_idx=1, index_tip_idx=13):
    """
    Preprocess hand landmarks:
      1. Normalize coordinates by image resolution.
      2. Recenter coordinates so that the palm (x1,y1) becomes (0,0).
      3. Scale coordinates using the Euclidean distance from palm to middle finger tip (x13,y13).
    """
    df_new = df.copy()
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] / img_width
        df_new[f'y{i}'] = df_new[f'y{i}'] / img_height

    palm_x = df_new[f'x{palm_idx}'].copy()
    palm_y = df_new[f'y{palm_idx}'].copy()
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] - palm_x
        df_new[f'y{i}'] = df_new[f'y{i}'] - palm_y
        
    scale = np.sqrt(df_new[f'x{index_tip_idx}']**2 + df_new[f'y{index_tip_idx}']**2) + 1e-8
    for i in range(1, num_points+1):
        df_new[f'x{i}'] = df_new[f'x{i}'] / scale
        df_new[f'y{i}'] = df_new[f'y{i}'] / scale    
    
    return df_new

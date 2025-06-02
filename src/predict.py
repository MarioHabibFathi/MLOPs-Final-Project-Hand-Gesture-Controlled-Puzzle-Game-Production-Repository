import cv2
import numpy as np
import pandas as pd
from collections import deque, Counter
import mediapipe as mp
import logging
import time
# from threading import Thread

logger = logging.getLogger(__name__)

# _latest_gesture = None


# def start_continuous_prediction(model, le, detector, camera_id=0, delay=1.0):
#     def run():
#         global _latest_gesture
#         cap = cv2.VideoCapture(camera_id)
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.warning("Failed to read frame from webcam.")
#                 time.sleep(delay)
#                 continue

#             try:
#                 gesture = predict(model, le, detector, frame)
#                 if gesture:
#                     _latest_gesture = gesture
#                     logger.info(f"Updated latest gesture to: {_latest_gesture}")
#             except Exception as e:
#                 logger.warning(f"Prediction error during continuous mode: {e}")

#             time.sleep(delay)

#     thread = Thread(target=run, daemon=True)
#     thread.start()



# def latest_gesture():
#     return _latest_gesture


def predict(model, le, detector, frame, delay=0):
    if delay > 0:
        time.sleep(delay)

    try:
        width = frame.shape[1]
        height = frame.shape[0]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #mp_image = mp.tasks.vision.Image(image_format=mp.tasks.vision.ImageFormat.SRGB, data=rgb_frame)
        #mp_image = Image(image_format=image_format.ImageFormat. SRGB,data=rgb_frame)
        results = detector.process(rgb_frame)
        # detection_result = detector.detect(results)
        # landmarks = detection_result.hand_landmarks[0]
        landmarks = results.multi_hand_landmarks

        if not landmarks:
            logger.warning("No hand detected.")
            return None

        # Only consider the first detected hand
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
        
        # Create a dictionary in the format x1...x21, y1...y21
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


# def predict(model, le, detector, frame, width, height, delay=0):
#     """
#     Predict hand gesture from a given frame.

#     Args:
#         model: Trained model
#         le: LabelEncoder
#         detector: MediaPipe hand detector
#         frame: Single video frame (BGR)
#         width: Frame width
#         height: Frame height
#         delay: Optional delay in seconds

#     Returns:
#         str or None: Predicted label or None if no hand detected
#     """
#     if delay > 0:
#         time.sleep(delay)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.tasks.vision.Image(image_format=mp.tasks.vision.ImageFormat.SRGB, data=rgb_frame)
#     detection_result = detector.detect(mp_image)

#     if not detection_result.hand_landmarks:
#         logger.warning("No hand detected.")
#         return None

#     landmarks = detection_result.hand_landmarks[0]

#     data = {'label': ['dummy']}
#     for i in range(21):
#         data[f'x{i+1}'] = [landmarks[i].x * width]
#         data[f'y{i+1}'] = [landmarks[i].y * height]

#     df = pd.DataFrame(data)
#     df_processed = preprocess_hand_landmarks(df, img_width=width, img_height=height)

#     features = []
#     for i in range(1, 22):
#         features.append(df_processed[f'x{i}'].values[0])
#         features.append(df_processed[f'y{i}'].values[0])

#     prediction = model.predict([features])[0]
#     label = le.inverse_transform([prediction])[0]

#     logger.info(f"Predicted gesture: {label}")
#     return label


# def predict(model, le, detector, delay=0, camera_index=0):
#     """
#     Capture one frame from webcam, detect hand gesture, and return predicted label.

#     Args:
#         model: Trained scikit-learn model for gesture classification
#         le: LabelEncoder used during training
#         detector: MediaPipe hand detector object
#         camera_index (int): Index of webcam device

#     Returns:
#         str: Predicted gesture label
#     """
    
#     if delay > 0:
#         time.sleep(delay)
    
#     cap = cv2.VideoCapture(camera_index)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     try:
#         ret, frame = cap.read()
#         if not ret:
#             logger.error("Failed to capture frame from camera.")
#             raise RuntimeError("Camera read failed.")

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         mp_image = mp.tasks.vision.Image(image_format=mp.tasks.vision.ImageFormat.SRGB, data=rgb_frame)
#         detection_result = detector.detect(mp_image)

#         if not detection_result.hand_landmarks:
#             logger.warning("No hand detected.")
#             return None

#         # Only use the first detected hand
#         landmarks = detection_result.hand_landmarks[0]

#         data = {'label': ['dummy']}
#         for i in range(21):
#             data[f'x{i+1}'] = [landmarks[i].x * width]
#             data[f'y{i+1}'] = [landmarks[i].y * height]

#         df = pd.DataFrame(data)
#         df_processed = preprocess_hand_landmarks(df, img_width=width, img_height=height)

#         features = []
#         for i in range(1, 22):
#             features.append(df_processed[f'x{i}'].values[0])
#             features.append(df_processed[f'y{i}'].values[0])

#         prediction = model.predict([features])[0]
#         label = le.inverse_transform([prediction])[0]

#         logger.info(f"Predicted gesture: {label}")
#         return label

#     except Exception as e:
#         logger.error(f"Gesture prediction failed: {str(e)}")
#         raise ValueError(f"Prediction error: {str(e)}")

#     finally:
#         cap.release()

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







# def predict(model, le, detector, camera_index=0, max_hands=1, history_length=15):
#     cap = cv2.VideoCapture(camera_index)
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     prediction_history = [deque(maxlen=history_length) for _ in range(max_hands)]
    
#     logger.info("Starting real-time gesture prediction...")

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 logger.warning("Failed to read frame from camera.")
#                 break

#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.tasks.vision.Image(image_format=mp.tasks.vision.ImageFormat.SRGB, data=rgb_frame)
#             detection_result = detector.detect(mp_image)

#             current_preds = [None] * max_hands

#             if detection_result.hand_landmarks:
#                 for hand_idx, landmarks in enumerate(detection_result.hand_landmarks[:max_hands]):
#                     data = {'label': ['dummy']}
#                     for i in range(21):
#                         data[f'x{i+1}'] = [landmarks[i].x * width]
#                         data[f'y{i+1}'] = [landmarks[i].y * height]
                    
#                     df = pd.DataFrame(data)

#                     df_processed = preprocess_hand_landmarks(df, img_width=width, img_height=height)
                    
#                     features = []
#                     for i in range(1, 22):
#                         features.append(df_processed[f'x{i}'].values[0])
#                         features.append(df_processed[f'y{i}'].values[0])

#                     pred = model.predict([features])[0]
#                     current_preds[hand_idx] = pred
#                     prediction_history[hand_idx].append(pred)

#                     mp.solutions.drawing_utils.draw_landmarks(
#                         frame,
#                         mp.solutions.hands.HandLandmarkList(
#                             landmark=[mp.solutions.framework.formats.landmark_pb2.NormalizedLandmark(
#                                 x=lm.x, y=lm.y, z=lm.z
#                             ) for lm in landmarks]
#                         ),
#                         mp.solutions.hands.HAND_CONNECTIONS
#                     )

#             # Stabilize predictions
#             for hand_idx in range(max_hands):
#                 history = prediction_history[hand_idx]
#                 if history:
#                     most_common = Counter(history).most_common(1)[0][0]
#                     gesture_name = le.inverse_transform([most_common])[0]
#                     cv2.putText(frame, f"Hand {hand_idx+1}: {gesture_name}", (10, 40 + 40 * hand_idx),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
#                     logger.debug(f"Predicted: {gesture_name}")

#             cv2.imshow("Live Gesture Prediction", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 logger.info("Quitting prediction loop.")
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         logger.info("Released video resources.")

# import pandas as pd
# import logging

# logger = logging.getLogger(__name__)

# def predict(model, preprocessor, input_data: dict):
#     """
#     Predict churn based on input data using pre-fitted model and preprocessor.

#     Args:
#         model: Trained scikit-learn model
#         preprocessor: Fitted ColumnTransformer
#         input_data: Raw input features (dict)

#     Returns:
#         int: Prediction (0 or 1)
#     """
#     df = pd.DataFrame([input_data])
#     logger.info(f"Input data: {input_data}")

#     try:
#         # Transform input using preprocessor
#         df_transformed = preprocessor.transform(df)

#         # Predict
#         prediction = model.predict(df_transformed)[0]
#         logger.info(f"Prediction successful: {prediction}")
#         return int(prediction)

#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise ValueError(f"Prediction error: {str(e)}")
        
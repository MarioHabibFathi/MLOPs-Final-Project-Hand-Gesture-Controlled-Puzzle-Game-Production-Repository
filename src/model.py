import joblib
import mediapipe as mp
def load_model():
    model = joblib.load("src/best_model.pkl")
    label_encoder = joblib.load("src/label_encoder.pkl")
    # detector = mp.tasks.vision.HandLandmarker.create_from_options(
    #     mp.tasks.vision.HandLandmarkerOptions(
    #         base_options=mp.tasks.BaseOptions(model_asset_path='src/hand_landmarker.task'),
    #         num_hands=1
    #     )
    # )
    detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    return model, label_encoder, detector

import pytest
from fastapi.testclient import TestClient
from api.main import app
from tests.test_landmark_data import DISLIKE_LANDMARKS
import json
client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Hand Gesture API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


# def test_dislike_gesture_prediction():
#     """Test that the 'dislike' gesture is correctly predicted"""
#     payload = {
#         "landmarks": DISLIKE_LANDMARKS
#     }
    
#     response = client.post("/predict-landmark", json=payload)
    
#     # Basic response validation
#     assert response.status_code == 200
#     response_data = response.json()
#     assert "gesture" in response_data
    
#     # Validate the prediction
#     assert response_data["gesture"] == "dislike", \
#         f"Expected 'dislike' gesture but got '{response_data['gesture']}'"

# def test_dislike_gesture_prediction():
#     """End-to-end test: API call → preprocessing → model prediction → response"""
#     # 1. Prepare test data (from your dataset)
#     payload = {
#         "landmarks": DISLIKE_LANDMARKS  # Your 21 landmarks data
#     }
    
#     # 2. Call the real API endpoint
#     response = client.post(
#         "/predict-landmark",
#         json=payload,
#         timeout=5.0  # Fail if API hangs
#     )
    
#     # 3. Verify HTTP layer worked
#     assert response.status_code == 200, \
#         f"API failed with status {response.status_code}. Response: {response.text}"
    
#     # 4. Verify response structure
#     response_data = response.json()
#     assert "gesture" in response_data, \
#         f"Missing 'gesture' key in response. Full response: {response_data}"
    
#     # 5. Verify business logic
#     assert response_data["gesture"] == "dislike", (
#         f"Prediction mismatch. Expected 'dislike', got '{response_data['gesture']}'. "
#         "Possible issues:\n"
#         "- Model was retrained with different labels\n"
#         "- Preprocessing changed\n"
#         "- Test data doesn't match training data distribution"
#     )
    
    
    
def test_dislike_gesture_prediction():
    """End-to-end test of the prediction API with debug on failure"""
    # 1. Prepare test data
    payload = {
        "landmarks": DISLIKE_LANDMARKS  # Your 21-landmark test data
    }
    print("[TEST DEBUG] Starting dislike test")

    # 2. Make API call
    response = client.post(
        "/predict-landmark",
        json=payload,
        timeout=5.0
    )
    
    print(f"[TEST DEBUG] Response: {response.status_code}, Body: {response.text}")
    # 3. Verify response
    try:
        assert response.status_code == 200, "API returned non-200 status"
        response_data = response.json()
        assert "gesture" in response_data, "Response missing 'gesture' field"
        assert response_data["gesture"] == "dislike", "Incorrect prediction"
        
    except AssertionError as e:
        # Debug output only shown on failure
        print("\n=== DEBUG INFO ===")
        print("Test failed at assertion:", str(e))
        print("\nRequest Payload:")
        print(json.dumps(payload, indent=2))
        
        print("\nAPI Response:")
        print(f"Status: {response.status_code}")
        print("Body:", json.dumps(response.json(), indent=2) if response.content else "Empty response")
        
        if response.status_code == 200 and "gesture" in response.json():
            print("Prediction Analysis:")
            print("Expected: 'dislike'")
            print(f"Received: '{response.json()['gesture']}'")
            
        raise  # Re-raise the assertion error
    
    
def test_predict_landmark_invalid_count():
    """Test validation for incorrect landmark count"""
    test_cases = [
        ([{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(20)], "20 landmarks"),  # Under
        ([{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(22)], "22 landmarks")   # Over
    ]
    
    for landmarks, description in test_cases:
        response = client.post("/predict-landmark", json={"landmarks": landmarks})
        assert response.status_code == 500, f"Failed for {description}"

def test_predict_landmark_missing_coords():
    """Test validation for malformed landmarks"""
    test_cases = [
        ([{"x": 0.1, "y": 0.2} for _ in range(21)], "missing z"),              # Missing coordinate
        ([{"x": "invalid", "y": 0.2, "z": 0.3} for _ in range(21)], "bad type"),  # Wrong type
        ([{"x": None, "y": 0.2, "z": 0.3} for _ in range(21)], "null value")     # Null value
    ]
    
    for landmarks, description in test_cases:
        response = client.post("/predict-landmark", json={"landmarks": landmarks})
        assert response.status_code == 422, f"Failed for {description}"

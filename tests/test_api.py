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


def test_dislike_gesture_prediction():
    """End-to-end test of the prediction API with debug on failure"""
    payload = {
        "landmarks": DISLIKE_LANDMARKS  
    }
    print("[TEST DEBUG] Starting dislike test")

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
        ([{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(20)], "20 landmarks"),  
        ([{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(22)], "22 landmarks")   
    ]
    
    for landmarks, description in test_cases:
        response = client.post("/predict-landmark", json={"landmarks": landmarks})
        assert response.status_code == 500, f"Failed for {description}"

def test_predict_landmark_missing_coords():
    """Test validation for malformed landmarks"""
    test_cases = [
        ([{"x": 0.1, "y": 0.2} for _ in range(21)], "missing z"),              
        ([{"x": "invalid", "y": 0.2, "z": 0.3} for _ in range(21)], "bad type"),  
        ([{"x": None, "y": 0.2, "z": 0.3} for _ in range(21)], "null value")     
    ]
    
    for landmarks, description in test_cases:
        response = client.post("/predict-landmark", json={"landmarks": landmarks})
        assert response.status_code == 422, f"Failed for {description}"

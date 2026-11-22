import cv2
from crowd_detection.config import load_config
from crowd_detection.main import get_available_camera

config = load_config("config.yaml")
cap, dev = get_available_camera(config.camera.device_index, config.camera.api_preference)
print("Device:", dev)
print("Backend name:", cap.getBackendName())
cap.release()

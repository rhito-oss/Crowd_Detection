import sys
import time
import cv2

from crowd_detection.config import load_config
from crowd_detection.density import evaluate_density
from crowd_detection.detector import PersonDetector
from crowd_detection.visualizer import draw_hud


def main():
    config = load_config("config.yaml")
    cap = cv2.VideoCapture(config.camera.device_index, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"Error: Unable to open webcam at index {config.camera.device_index}.", file=sys.stderr)
        print("Please check camera permissions, connection, or /dev/video* permissions.", file=sys.stderr)
        sys.exit(1)

    # Request high resolution; OpenCV and camera drivers automatically negotiate down to the camera's highest native capability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam initialized successfully ({actual_w}x{actual_h} @ {actual_fps:.1f} FPS).")

    print(f"Loading person detector ({config.detection.model} on {config.detection.device})...")
    detector = PersonDetector(
        model_name=config.detection.model,
        conf_threshold=config.detection.conf_threshold,
        iou_threshold=config.detection.iou_threshold,
        device=config.detection.device,
    )

    window_name = config.display.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to grab frame from webcam. Retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            density_result = evaluate_density(detections, w, h, config.density)

            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff) if fps > 0 else (1.0 / time_diff)
            prev_time = curr_time

            display_frame = draw_hud(frame, detections, density_result, fps, config.display)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Exit signal received. Closing...")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user. Exiting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")

if __name__ == "__main__":
    main()

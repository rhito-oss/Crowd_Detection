import sys
import time
import cv2

from crowd_detection.audio import AudioAlert
from crowd_detection.config import load_config
from crowd_detection.density import evaluate_density
from crowd_detection.detector import PersonDetector
from crowd_detection.stampede import calculate_stampede_probability
from crowd_detection.visualizer import draw_hud


def get_available_camera(preferred_index: int = 0, preferred_api: str = "dshow"):
    api_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }

    if preferred_api.lower() in ("dshow", "default", "any") and sys.platform.startswith("win"):
        api_backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif preferred_api.lower() in api_map:
        api_backends = [api_map[preferred_api.lower()]]
    else:
        api_backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]

    if cv2.CAP_DSHOW not in api_backends and sys.platform.startswith("win"):
        api_backends.append(cv2.CAP_DSHOW)
    if cv2.CAP_ANY not in api_backends:
        api_backends.append(cv2.CAP_ANY)

    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]

    for api in api_backends:
        for idx in indices:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    return cap, idx
            cap.release()

    return None, -1


def main():
    config = load_config("config.yaml")
    cap, device_id = get_available_camera(
        config.camera.device_index,
        getattr(config.camera, "api_preference", "any"),
    )

    if cap is None:
        print("Error: No available webcam found on any device index.", file=sys.stderr)
        print("Please check camera connection and permissions.", file=sys.stderr)
        sys.exit(1)

    print(f"Connected to camera on device index: {device_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    audio_alert = AudioAlert(
        audio_file=config.audio.file_path,
        cooldown=config.audio.cooldown,
    )
    crowd_start_time = None

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            try:
                ret, frame = cap.read()
            except cv2.error as e:
                print(f"Warning: OpenCV error during frame capture: {e}", file=sys.stderr)
                time.sleep(0.05)
                continue

            if not ret or frame is None or frame.size == 0:
                print("Warning: Failed to grab frame from webcam. Retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            curr_time = time.time()
            detections, avg_speed = detector.detect(frame, timestamp=curr_time)
            density_result = evaluate_density(detections, w, h, config.density)
            stampede_result = calculate_stampede_probability(
                detections=detections,
                density_result=density_result,
                avg_speed_kmh=avg_speed,
                config=config.stampede,
            )

            if config.audio.enabled and density_result.is_crowded:
                if crowd_start_time is None:
                    crowd_start_time = curr_time
                elif curr_time - crowd_start_time >= config.audio.persist_duration:
                    audio_alert.play()
            else:
                crowd_start_time = None

            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_diff) if fps > 0 else (1.0 / time_diff)
            prev_time = curr_time

            display_frame = draw_hud(
                frame,
                detections,
                density_result,
                fps,
                config.display,
                avg_speed=avg_speed,
                stampede_result=stampede_result,
            )

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

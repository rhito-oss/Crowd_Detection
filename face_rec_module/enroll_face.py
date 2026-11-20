import argparse
import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from face_rec_module.face_database import TwoTierFaceDatabase


def get_available_camera(preferred_index: int = 0):
    """Finds the first functional camera device and breaks immediately."""
    indices = [preferred_index] + [i for i in range(5) if i != preferred_index]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, idx
        cap.release()
    return None, -1


def main():
    parser = argparse.ArgumentParser(description="High-Speed Face Enrollment Scanner")
    parser.add_argument("--name", type=str, default="", help="Name of the person to enroll")
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        try:
            name = input("Enter person's name to enroll: ").strip()
        except EOFError:
            name = "User"
    if not name:
        name = "User"

    print(f"\n[ENROLLMENT INITIALIZING] Profile Name: {name}")
    print("Loading High-Speed CPU model (buffalo_sc @ 320x320)...")

    app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))

    db = TwoTierFaceDatabase()
    cap, cam_idx = get_available_camera(0)

    if cap is None:
        print("Error: Could not access any webcam.", file=sys.stderr)
        sys.exit(1)

    print(f"Webcam connected on device index {cam_idx}.")
    print("\n================ INSTRUCTIONS ================")
    print("1. Center your face inside the green guide reticle.")
    print("2. For EACH of the 5 angles, position your face and PRESS SPACE when ready.")
    print("3. Angle steps: Center -> Turn Left -> Turn Right -> Tilt Up -> Tilt Down.")
    print("4. Press Q or ESC to cancel.")
    print("==============================================\n")

    burst_count = 5
    captured_embeddings = []
    flash_feedback_timer = 0.0

    prompts = [
        "Angle 1/5: Look Straight Center -> [PRESS SPACE]",
        "Angle 2/5: Turn Head Slightly Left -> [PRESS SPACE]",
        "Angle 3/5: Turn Head Slightly Right -> [PRESS SPACE]",
        "Angle 4/5: Tilt Head Slightly Up -> [PRESS SPACE]",
        "Angle 5/5: Tilt Head Slightly Down -> [PRESS SPACE]",
    ]

    window_name = f"Face Enrollment Scanner - [{name}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            display = frame.copy()

            # Fast CPU detection
            faces = app.get(frame)
            best_face = None
            if faces:
                best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            now = time.time()
            is_flashing = (now - flash_feedback_timer) < 0.3

            cx, cy = w // 2, h // 2
            rx, ry = int(w * 0.18), int(h * 0.28)
            reticle_color = (0, 255, 255) if is_flashing else ((0, 229, 255) if best_face is not None else (0, 0, 255))

            cv2.ellipse(display, (cx, cy), (rx, ry), 0, 0, 360, reticle_color, 3 if is_flashing else 2)
            cv2.circle(display, (cx, cy), 4, reticle_color, -1)

            if best_face is not None:
                bbox = best_face.bbox.astype(int)
                box_color = (0, 255, 255) if is_flashing else (0, 229, 255)
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 1)

                if hasattr(best_face, "kps") and best_face.kps is not None:
                    pts = best_face.kps.astype(int)
                    # Semi-transparent triangulation mesh
                    overlay = display.copy()
                    for p1, p2 in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]:
                        cv2.line(overlay, tuple(pts[p1]), tuple(pts[p2]), (0, 229, 255), 1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

                    for pt in pts:
                        cv2.circle(display, tuple(pt), 3, (255, 255, 255), -1)

            step_idx = len(captured_embeddings)
            current_prompt = prompts[step_idx] if step_idx < burst_count else "Processing..."

            cv2.rectangle(display, (0, 0), (w, 64), (15, 20, 28), -1)
            cv2.line(display, (0, 64), (w, 64), (0, 229, 255), 1)

            cv2.putText(
                display,
                f"PROFILE: {name.upper()} | CAPTURED: {step_idx}/{burst_count}",
                (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 229, 255),
                2,
            )
            cv2.putText(
                display,
                current_prompt,
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128) if best_face is not None else (0, 140, 255),
                1,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if best_face is not None and best_face.embedding is not None:
                    captured_embeddings.append(best_face.embedding.copy())
                    flash_feedback_timer = time.time()
                    print(f"[CAPTURED {len(captured_embeddings)}/{burst_count}] {prompts[step_idx]}")

                    if len(captured_embeddings) >= burst_count:
                        print("\nAll 5 angles captured! Storing multi-vector profile gallery...")
                        # Save each individual angle vector + blended mean vector into FAISS
                        # so FAISS matches you directly whether you are looking center, left, right, up, or down!
                        master_vec = np.mean(captured_embeddings, axis=0)
                        all_vectors = captured_embeddings + [master_vec]
                        assigned_id = db.register_identity_multi_vector(name=name, embeddings=all_vectors)

                        print(f"==================================================")
                        print(f"SUCCESS: Enrolled '{name}' with 6-Vector Gallery!")
                        print(f"Assigned ID: #{assigned_id}")
                        print(f"FAISS Total Stored Vectors: {db.get_total_identities()}")
                        print(f"==================================================")
                        break
                else:
                    print("No clear face detected! Align your face inside the reticle and press SPACE.")

            elif key == ord("q") or key == 27:
                print("Enrollment cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

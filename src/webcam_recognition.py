
"""Real-time face recognition from webcam - SIMPLE VERSION (fixed: shows window)"""
import sys
import os
import cv2
import logging
import numpy as np
from face_recognition import BiometricFaceRecognition
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def webcam_recognition(model_path: str = DB_PICKLE):
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)

    logger.info("🚀 Loading pipeline...")
    pipeline = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    pipeline.load_enrollment_db(model_path)

    logger.info("📷 Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Could not open webcam")
        sys.exit(1)

    logger.info("✓ Webcam started")
    frame_count = 0
    detected_people = {}
    os.makedirs("output", exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ Failed to read frame from webcam")
                break

            # run detection/recognition every VIDEO_FRAME_SKIP frames
            if frame_count % VIDEO_FRAME_SKIP == 0:
                try:
                    enhanced = pipeline.enhancer.enhance_biometric_image(frame)
                    results = pipeline.detector(enhanced, conf=0.3)

                    if len(results) > 0 and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()

                        for box in boxes:
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                            # skip invalid crops
                            if x2 <= x1 or y2 <= y1:
                                continue

                            face_crop = enhanced[y1:y2, x1:x2]

                            try:
                                faces = pipeline.face_analysis.get(face_crop)
                                if len(faces) > 0:
                                    embedding = faces[0].embedding

                                    best_match = None
                                    best_prob = 0.0

                                    for person_id, enrolled_emb in pipeline.enrollment_db.items():
                                        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
                                        enrolled_norm = enrolled_emb / (np.linalg.norm(enrolled_emb) + 1e-6)
                                        similarity = float(np.dot(emb_norm, enrolled_norm))
                                        probability = (similarity + 1) / 2.0

                                        if probability > best_prob:
                                            best_prob = probability
                                            best_match = pipeline.person_names.get(person_id)

                                    name = best_match if best_prob >= RECOGNITION_THRESHOLD else "Unknown"
                                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, f"{name} ({best_prob:.2f})", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                                    # STOP ON FIRST RECOGNITION
                                    if name != "Unknown":
                                        detected_people[name] = detected_people.get(name, 0) + 1
                                        cv2.imwrite("output/webcam_frame.jpg", frame)

                                        logger.info("\n" + "=" * 50)
                                        logger.info("✓ FACE RECOGNIZED!")
                                        logger.info(f"  Name: {name}")
                                        logger.info(f"  Confidence: {best_prob:.4f}")
                                        logger.info(f"  Frame: {frame_count}")
                                        logger.info("  Saved to: output/webcam_frame.jpg")
                                        logger.info("=" * 50 + "\n")

                                        # clean up UI and camera then return
                                        cv2.imshow("Webcam - Face Recognition", frame)
                                        cv2.waitKey(100)  # brief pause so user can see the final frame
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        return
                            except Exception:
                                logger.debug("Face analysis failed for a crop", exc_info=True)

                except Exception:
                    logger.debug("Detection/enhancement failed on this frame", exc_info=True)

            # always save latest frame (optional)
            cv2.imwrite("output/webcam_frame.jpg", frame)

            # SHOW CAMERA WINDOW (this is why window appears)
            cv2.imshow("Webcam - Face Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            # press 'q' to quit
            if key == ord('q'):
                logger.info("q pressed -> exiting")
                break

            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count} | Searching...")

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("\n✓ Stopped by user")

    finally:
        # ensure resources are released
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logger.info(f"\n✓ Finished")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  People detected: {detected_people}")


if __name__ == "__main__":
    webcam_recognition()
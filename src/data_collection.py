"""
Script to collect face images from webcam for training
Usage: python data_collection.py --name "John_Doe" --count 10
"""

import cv2
import os
import argparse
from pathlib import Path


def collect_face_data(person_name: str, num_images: int = 10, output_dir: str = "known_faces"):
    """Capture face images from webcam"""
    
    # Create person directory
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"📷 Capturing {num_images} images for {person_name}")
    print("Instructions:")
    print("  - Face the camera")
    print("  - Press SPACE to capture image")
    print("  - Try different angles/lighting")
    print("  - Press Q to quit early")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display count
        cv2.putText(frame, f"Captured: {count}/{num_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow("Face Capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Save image
            filename = os.path.join(person_dir, f"{person_name}_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"✓ Saved image {count}/{num_images}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Completed! {count} images saved in {person_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect face images from webcam")
    parser.add_argument("--name", type=str, required=True, help="Person's name")
    parser.add_argument("--count", type=int, default=10, help="Number of images to capture")
    parser.add_argument("--output-dir", type=str, default="known_faces", help="Output directory")
    
    args = parser.parse_args()
    collect_face_data(args.name, args.count, args.output_dir)
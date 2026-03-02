"""
Register known faces into the database
This processes all images in known_faces/ directory
"""

import os
import cv2
from face_recognition import FaceRecognitionPipeline
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_all_faces(known_faces_dir: str = "known_faces", output_model: str = DATABASE_PATH):
    """
    Register all faces from directory structure
    Expected structure:
        known_faces/
        ├── person_1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── person_2/
            └── img1.jpg
    """
    
    # Initialize pipeline
    logger.info("🚀 Initializing Face Recognition Pipeline...")
    pipeline = FaceRecognitionPipeline(yolo_model=YOLO_MODEL, device=DETECTION_DEVICE)
    
    # Counter
    total_faces = 0
    failed = 0
    
    # Iterate through each person directory
    for person_name in sorted(os.listdir(known_faces_dir)):
        person_path = os.path.join(known_faces_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        logger.info(f"\n📝 Processing: {person_name}")
        
        # Iterate through images in person directory
        image_count = 0
        for image_file in sorted(os.listdir(person_path)):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(person_path, image_file)
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"  ⚠️  Could not read: {image_file}")
                    failed += 1
                    continue
                
                # Register face
                success = pipeline.register_face(image, person_name)
                if success:
                    image_count += 1
                    total_faces += 1
                    logger.info(f"  ✓ {image_file}")
                else:
                    failed += 1
                    logger.warning(f"  ✗ Failed to register: {image_file}")
            
            except Exception as e:
                logger.error(f"  ✗ Error processing {image_file}: {e}")
                failed += 1
        
        logger.info(f"  Summary: {image_count} faces registered for {person_name}")
    
    # Save model
    logger.info(f"\n💾 Saving model to {output_model}...")
    pipeline.save_model(output_model)
    
    # Final summary
    logger.info(f"\n" + "="*50)
    logger.info(f"✓ REGISTRATION COMPLETE")
    logger.info(f"  Total faces registered: {total_faces}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Model saved: {output_model}")
    logger.info(f"="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register known faces")
    parser.add_argument("--dir", type=str, default="known_faces", help="Known faces directory")
    parser.add_argument("--output", type=str, default=DATABASE_PATH, help="Output model path")
    
    args = parser.parse_args()
    register_all_faces(args.dir, args.output)
"""
Recognize faces in test images
"""

import sys
import os
import cv2
from face_recognition import FaceRecognitionPipeline
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recognize_single_image(image_path: str, model_path: str = DATABASE_PATH):
    """Recognize faces in a single image"""
    
    # Check if files exist
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Run: python register_known_faces.py")
        sys.exit(1)
    
    # Load pipeline
    logger.info("🚀 Loading pipeline...")
    pipeline = FaceRecognitionPipeline(yolo_model=YOLO_MODEL, device=DETECTION_DEVICE)
    pipeline.load_model(model_path)
    
    # Process image
    logger.info(f"📸 Processing: {image_path}")
    output_image, faces = pipeline.recognize_in_image(image_path, threshold=RECOGNITION_THRESHOLD)
    
    # Save output
    output_path = os.path.join("output", os.path.basename(image_path))
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, output_image)
    
    # Print results
    logger.info(f"\n✓ RECOGNITION COMPLETE")
    logger.info(f"  Total faces found: {len(faces)}")
    for i, face in enumerate(faces, 1):
        logger.info(f"\n  Face {i}:")
        logger.info(f"    Name: {face.name}")
        logger.info(f"    Confidence: {face.confidence:.4f}")
        logger.info(f"    Age: {face.age}")
        logger.info(f"    Gender: {face.gender}")
        logger.info(f"    Location: {face.bbox}")
    
    logger.info(f"\n  Output saved: {output_path}")


def recognize_all_images(test_dir: str = "test_images", model_path: str = DATABASE_PATH):
    """Recognize faces in all images in directory"""
    
    logger.info(f"🚀 Loading pipeline...")
    pipeline = FaceRecognitionPipeline(yolo_model=YOLO_MODEL, device=DETECTION_DEVICE)
    pipeline.load_model(model_path)
    
    os.makedirs("output", exist_ok=True)
    
    for image_file in sorted(os.listdir(test_dir)):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        image_path = os.path.join(test_dir, image_file)
        logger.info(f"\n📸 Processing: {image_file}")
        
        output_image, faces = pipeline.recognize_in_image(image_path, threshold=RECOGNITION_THRESHOLD)
        output_path = os.path.join("output", image_file)
        cv2.imwrite(output_path, output_image)
        
        logger.info(f"  Found {len(faces)} faces")
        for face in faces:
            logger.info(f"    - {face.name} (confidence: {face.confidence:.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognize faces in images")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--dir", type=str, default="test_images", help="Directory with test images")
    parser.add_argument("--model", type=str, default=DATABASE_PATH, help="Model path")
    
    args = parser.parse_args()
    
    if args.image:
        recognize_single_image(args.image, args.model)
    else:
        recognize_all_images(args.dir, args.model)
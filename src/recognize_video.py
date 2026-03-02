"""
Recognize faces in videos with real-time output
"""

import sys
import os
import cv2
from face_recognition import FaceRecognitionPipeline
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recognize_in_video(video_path: str, output_path: str = None, model_path: str = DATABASE_PATH):
    """Recognize faces in video"""
    
    # Check if files exist
    if not os.path.exists(video_path):
        logger.error(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Run: python register_known_faces.py")
        sys.exit(1)
    
    # Set default output
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"output/{video_name}_recognized.mp4"
    
    # Load pipeline
    logger.info("🚀 Loading pipeline...")
    pipeline = FaceRecognitionPipeline(yolo_model=YOLO_MODEL, device=DETECTION_DEVICE)
    pipeline.load_model(model_path)
    
    # Process video
    logger.info(f"🎬 Processing video: {video_path}")
    logger.info(f"   Output: {output_path}")
    logger.info("   Press 'q' to stop processing")
    
    pipeline.recognize_in_video(video_path, output_path=output_path, 
                               threshold=RECOGNITION_THRESHOLD)
    
    logger.info(f"\n✓ Video processing complete!")
    logger.info(f"  Output saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Recognize faces in videos")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--model", type=str, default=DATABASE_PATH, help="Model path")
    
    args = parser.parse_args()
    recognize_in_video(args.video, args.output, args.model)
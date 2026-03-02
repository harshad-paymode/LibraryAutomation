"""
Prepare biometric enrollment data
Convert image files to standardized format
"""

import cv2
import os
import sys
from pathlib import Path

def prepare_enrollment_images(source_dir: str, output_dir: str = "database/"):
    """
    Prepare enrollment images
    Standardizes format and checks quality
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in sorted(os.listdir(source_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        source_path = os.path.join(source_dir, filename)
        
        # Read image
        image = cv2.imread(source_path)
        if image is None:
            print(f"❌ Failed to read: {filename}")
            continue
        
        # Check size (minimum 100x100)
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            print(f"⚠️  Too small ({w}x{h}): {filename}")
            continue
        
        # Resize to standard height (keeping aspect ratio)
        if h > 600:
            scale = 600 / h
            new_h = 600
            new_w = int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Save to database directory
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"✓ {filename} -> {output_filename}")
    
    print(f"\n✓ Prepared images in: {output_dir}")

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "raw_faces/"
    prepare_enrollment_images(source)
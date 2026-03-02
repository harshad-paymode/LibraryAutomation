"""Configuration for Biometric Face Recognition"""

# YOLO Settings
YOLO_MODEL = "yolo26n.pt"

# Device Settings
DEVICE = 0  # GPU device ID (0 for first GPU, -1 for CPU)

# Biometric Recognition Settings
ENROLLMENT_THRESHOLD = 0.60   # For verification (0.55-0.65 recommended)
MIN_QUALITY_SCORE = 0.5       # Minimum acceptable face quality

# Paths
DATABASE_DIR = "LibraryAutomation/faceData/"          # Enrollment images directory
TEST_DIR = "LibraryAutomation/faceData/"           # Test images directory
OUTPUT_DIR = "LibraryAutomation/output/"              # Output directory
DB_PICKLE = "C:/Users/paymo/Downloads/LibraryAutomation/models/face_embeddings_db.pkl"

# Logging
LOG_FILE = "logs/biometric_recognition.log"

# Recognition Settings
RECOGNITION_THRESHOLD = 0.75  # Match threshold (0.55-0.65 recommended)

# Video/Webcam Settings
VIDEO_FRAME_SKIP = 1  # Process every 1st frame (increase for speed, e.g., 3 = every 3rd frame)
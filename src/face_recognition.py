"""
Optimized Biometric Face Recognition Pipeline
Single front-face enrollment per person
Maximum accuracy with minimal training data
"""

import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pickle
import os
from pathlib import Path
import csv
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BiometricMatch:
    """Biometric match result"""
    person_id: str
    match_probability: float  # 0-1 (1 = perfect match)
    similarity_score: float   # cosine similarity (-1 to 1)
    matched: bool             # Above threshold
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ImageEnhancer:
    """Image enhancement specifically for frontal face biometrics"""
    
    @staticmethod
    def enhance_biometric_image(image: np.ndarray) -> np.ndarray:
        """
        Optimize frontal face image for biometric recognition
        Focus on face quality improvement, not excessive enhancement
        """
        enhanced = image.copy()
        
        # 1. Denoise (gentle, preserve details)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # 2. CLAHE for contrast (careful application)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Slight sharpening for frontal faces
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]]) / 1.5
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced


class FaceQualityAssessor:
    """Assess quality of detected face for biometric purposes"""
    
    @staticmethod
    def assess_quality(face_image: np.ndarray, kps: np.ndarray = None) -> Tuple[float, str]:
        """
        Assess face image quality for biometric recognition
        Returns: (quality_score: 0-1, reason: str)
        """
        h, w = face_image.shape[:2]
        quality_score = 1.0
        reasons = []
        
        # 1. Check face size (minimum requirements)
        if h < 80 or w < 80:
            quality_score -= 0.3
            reasons.append(f"Face too small ({w}x{h})")
        
        # 2. Check for blur (Laplacian variance)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:  # Very blurry
            quality_score -= 0.4
            reasons.append("Image too blurry")
        elif blur_score < 100:  # Somewhat blurry
            quality_score -= 0.15
            reasons.append("Image slightly blurry")
        
        # 3. Check brightness (face too dark/bright)
        brightness = np.mean(gray)
        if brightness < 30:  # Too dark
            quality_score -= 0.2
            reasons.append("Face too dark")
        elif brightness > 225:  # Too bright
            quality_score -= 0.2
            reasons.append("Face overexposed")
        
        # 4. Check contrast
        contrast = np.std(gray)
        if contrast < 15:
            quality_score -= 0.15
            reasons.append("Low contrast")
        
        quality_score = max(0.0, min(1.0, quality_score))
        reason = " | ".join(reasons) if reasons else "Good quality"
        
        return quality_score, reason
    
    @staticmethod
    def is_frontal_face(face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if face is frontal (not too tilted)
        Returns: (is_frontal: bool, head_pose_score: 0-1)
        
        Uses symmetry analysis for frontal face detection
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Split face vertically
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        
        # Calculate symmetry (frontal faces are more symmetric)
        if left_half.shape == right_half.shape:
            symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF)
            frontality_score = np.mean(symmetry)
        else:
            frontality_score = 0.5
        
        # Normalize score to 0-1
        frontality_score = np.clip(frontality_score / 100.0, 0, 1)
        
        is_frontal = frontality_score > 0.4  # Threshold for frontal face
        
        return is_frontal, frontality_score


class BiometricFaceRecognition:
    """
    Biometric Face Recognition System
    Optimized for single enrollment face per person
    """
    
    def __init__(self, yolo_model: str = "yolo26n.pt", device: int = 0):
        """
        Initialize biometric recognition system
        
        Args:
            yolo_model: Path to YOLO face detector
            device: GPU device ID (-1 for CPU)
        """
        logger.info("🚀 Initializing Biometric Face Recognition System...")
        
        self.detector = YOLO(yolo_model)
        self.face_analysis = FaceAnalysis(name="buffalo_l")
        self.face_analysis.prepare(ctx_id=device, det_size=(640, 640))
        
        self.enhancer = ImageEnhancer()
        self.quality_assessor = FaceQualityAssessor()
        
        # Database: person_id -> embedding
        self.enrollment_db: Dict[str, np.ndarray] = {}
        self.person_names: Dict[str, str] = {}  # person_id -> display_name
        
        logger.info("✓ System initialized successfully")
    
    def extract_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract face embedding from image with quality assessment
        
        Returns:
            (embedding: np.ndarray, metadata: Dict)
        """
        # Detect face
        results = self.detector(image, conf=0.3)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, {'error': 'No face detected'}
        
        # Get largest face (should be only one for biometric)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_box_idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[largest_box_idx]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Crop and enhance
        face_crop = image[y1:y2, x1:x2]
        enhanced_crop = self.enhancer.enhance_biometric_image(face_crop)
        
        # Assess quality
        quality_score, quality_reason = self.quality_assessor.assess_quality(face_crop)
        is_frontal, frontality_score = self.quality_assessor.is_frontal_face(face_crop)
        
        # Get embedding
        try:
            faces = self.face_analysis.get(enhanced_crop)
            if len(faces) == 0:
                return None, {'error': 'No face recognized by InsightFace'}
            
            embedding = faces[0].embedding
            
            metadata = {
                'bbox': (x1, y1, x2, y2),
                'quality_score': float(quality_score),
                'quality_reason': quality_reason,
                'is_frontal': bool(is_frontal),
                'frontality_score': float(frontality_score),
                'age': faces[0].age,
                'gender': 'M' if faces[0].gender == 0 else 'F',
            }
            
            return embedding, metadata
        
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None, {'error': str(e)}
    
    def enroll_face(self, image_path: str, person_id: str, person_name: str = None) -> bool:
        """
        Enroll single face for a person
        
        Args:
            image_path: Path to enrollment image
            person_id: Unique person identifier (e.g., "P001", "john_doe")
            person_name: Display name (optional, defaults to person_id)
        
        Returns:
            bool: Success status
        """
        logger.info(f"📝 Enrolling: {person_id} ({person_name or person_id})")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Could not read image: {image_path}")
            return False
        
        # Extract embedding
        embedding, metadata = self.extract_embedding(image)
        if embedding is None:
            logger.error(f"❌ Failed to extract embedding: {metadata.get('error')}")
            return False
        
        # Check quality
        if metadata['quality_score'] < 0.6:
            logger.warning(f"⚠️  Low quality face ({metadata['quality_reason']})")
            logger.warning("   Consider using better quality enrollment image")
        
        if not metadata['is_frontal']:
            logger.warning(f"⚠️  Face not frontal (frontality: {metadata['frontality_score']:.2f})")
            logger.warning("   For best accuracy, use frontal face")
        
        # Store enrollment
        self.enrollment_db[person_id] = embedding
        self.person_names[person_id] = person_name or person_id
        
        logger.info(f"✓ Enrolled successfully")
        logger.info(f"  Quality: {metadata['quality_score']:.2f} | {metadata['quality_reason']}")
        logger.info(f"  Frontality: {metadata['frontality_score']:.2f}")
        logger.info(f"  Age: {metadata['age']}, Gender: {metadata['gender']}")
        
        return True
    
    def verify_face(self, image_path: str, threshold: float = 0.6) -> List[BiometricMatch]:
        """
        Verify/recognize face against enrolled database
        
        Args:
            image_path: Path to verification image
            threshold: Similarity threshold (0.5-0.7 recommended for biometric)
        
        Returns:
            List of matches sorted by probability (descending)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Could not read image: {image_path}")
            return []
        
        # Extract embedding
        embedding, metadata = self.extract_embedding(image)
        if embedding is None:
            logger.error(f"❌ Failed to extract embedding: {metadata.get('error')}")
            return []
        
        # Compare with enrolled faces
        matches = []
        for person_id, enrolled_embedding in self.enrollment_db.items():
            # Normalize embeddings
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
            enrolled_norm = enrolled_embedding / (np.linalg.norm(enrolled_embedding) + 1e-6)
            
            # Cosine similarity
            similarity = np.dot(emb_norm, enrolled_norm)
            
            # Convert similarity (-1 to 1) to probability (0 to 1)
            probability = (similarity + 1) / 2  # Maps [-1, 1] to [0, 1]
            
            # Determine match
            matched = probability >= threshold
            
            match = BiometricMatch(
                person_id=person_id,
                match_probability=probability,
                similarity_score=similarity,
                matched=matched
            )
            matches.append(match)
        
        # Sort by probability (descending)
        matches.sort(key=lambda x: x.match_probability, reverse=True)
        
        return matches
    
    def verify_face_detailed(self, image_path: str, threshold: float = 0.6) -> Dict:
        """Get detailed verification report"""
        image = cv2.imread(image_path)
        embedding, metadata = self.extract_embedding(image)
        
        if embedding is None:
            return {
                'success': False,
                'error': metadata.get('error'),
                'image': image_path
            }
        
        matches = self.verify_face(image_path, threshold)
        top_match = matches[0] if matches else None
        
        return {
            'success': True,
            'image': image_path,
            'metadata': metadata,
            'top_match': {
                'person_id': top_match.person_id,
                'person_name': self.person_names.get(top_match.person_id),
                'probability': top_match.match_probability,
                'similarity': top_match.similarity_score,
                'matched': top_match.matched,
                'timestamp': top_match.timestamp
            } if top_match else None,
            'all_matches': [
                {
                    'person_id': m.person_id,
                    'person_name': self.person_names.get(m.person_id),
                    'probability': m.match_probability,
                    'similarity': m.similarity_score,
                    'matched': m.matched
                }
                for m in matches
            ]
        }
    
    def save_enrollment_db(self, filepath: str) -> bool:
        """Save enrollment database"""
        try:
            data = {
                'enrollment_db': self.enrollment_db,
                'person_names': self.person_names
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"💾 Database saved: {filepath}")
            logger.info(f"   Enrolled: {len(self.enrollment_db)} people")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving database: {e}")
            return False
    
    def load_enrollment_db(self, filepath: str) -> bool:
        """Load enrollment database"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.enrollment_db = data['enrollment_db']
            self.person_names = data['person_names']
            logger.info(f"📂 Database loaded: {filepath}")
            logger.info(f"   Enrolled: {len(self.enrollment_db)} people")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading database: {e}")
            return False
    
    def batch_enroll(self, database_dir: str) -> Dict:
        """
        Batch enroll faces from directory
        Filename = person_id (e.g., "person_001.jpg", "john_doe.jpg")
        """
        logger.info(f"📁 Batch enrolling from: {database_dir}")
        
        stats = {'success': 0, 'failed': 0, 'low_quality': 0}
        
        for filename in sorted(os.listdir(database_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # Extract person_id from filename (without extension)
            person_id = os.path.splitext(filename)[0]
            image_path = os.path.join(database_dir, filename)
            
            if self.enroll_face(image_path, person_id):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"\n✓ Batch enrollment complete:")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        
        return stats
    
    def batch_verify(self, test_dir: str, threshold: float = 0.6) -> List[Dict]:
        """
        Batch verify faces from test directory
        """
        logger.info(f"🔍 Batch verifying from: {test_dir}")
        logger.info(f"   Threshold: {threshold}")
        
        results = []
        
        for filename in sorted(os.listdir(test_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(test_dir, filename)
            logger.info(f"\n📸 {filename}")
            
            report = self.verify_face_detailed(image_path, threshold)
            results.append(report)
            
            if report['success']:
                top = report['top_match']
                if top and top['matched']:
                    logger.info(f"  ✓ MATCH: {top['person_name']} ({top['probability']:.4f})")
                else:
                    logger.info(f"  ✗ NO MATCH (top: {top['person_name']} - {top['probability']:.4f})")
            else:
                logger.info(f"  ✗ ERROR: {report['error']}")
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str):
        """Generate verification report"""
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Top_Match', 'Probability', 'Matched', 'Quality', 'Status'])
            
            for result in results:
                if result['success']:
                    top = result['top_match']
                    writer.writerow([
                        os.path.basename(result['image']),
                        top['person_name'] if top else 'N/A',
                        f"{top['probability']:.4f}" if top else 'N/A',
                        'YES' if (top and top['matched']) else 'NO',
                        f"{result['metadata']['quality_score']:.2f}",
                        'SUCCESS'
                    ])
                else:
                    writer.writerow([
                        os.path.basename(result['image']),
                        'N/A',
                        'N/A',
                        'N/A',
                        'N/A',
                        f"FAILED: {result['error']}"
                    ])
        
        logger.info(f"📊 Report saved: {output_file}")
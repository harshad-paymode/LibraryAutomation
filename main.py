"""
Biometric Face Recognition - Main Entry Point
Simplified for single face per person enrollment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from src.face_recognition import BiometricFaceRecognition
from src.config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    for directory in [DATABASE_DIR, TEST_DIR, OUTPUT_DIR, "models", "logs"]:
        os.makedirs(directory, exist_ok=True)


def cmd_enroll(args):
    """Enroll single face"""
    system = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    
    # Load existing database if available
    if os.path.exists(DB_PICKLE):
        system.load_enrollment_db(DB_PICKLE)
    
    success = system.enroll_face(args.image, args.id, args.name)
    
    if success:
        system.save_enrollment_db(DB_PICKLE)
        logger.info(f"✓ Total enrolled: {len(system.enrollment_db)}")
    
    return 0 if success else 1


def cmd_batch_enroll(args):
    """Batch enroll from directory"""
    system = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    
    # Load existing database if available
    if os.path.exists(DB_PICKLE):
        system.load_enrollment_db(DB_PICKLE)
    
    stats = system.batch_enroll(args.dir)
    system.save_enrollment_db(DB_PICKLE)
    
    return 0 if stats['failed'] == 0 else 1


def cmd_verify_single(args):
    """Verify single image"""
    system = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    
    if not os.path.exists(DB_PICKLE):
        logger.error("❌ No enrollment database found. Run enrollment first.")
        return 1
    
    system.load_enrollment_db(DB_PICKLE)
    
    threshold = args.threshold if args.threshold else ENROLLMENT_THRESHOLD
    report = system.verify_face_detailed(args.image, threshold)
    
    # Print results
    if report['success']:
        print(f"\n{'='*60}")
        print(f"VERIFICATION RESULT")
        print(f"{'='*60}")
        print(f"Image: {os.path.basename(report['image'])}")
        print(f"Quality: {report['metadata']['quality_score']:.2f}")
        print(f"Frontal: {report['metadata']['is_frontal']}")
        
        if report['top_match']:
            top = report['top_match']
            print(f"\nTop Match: {top['person_name']}")
            print(f"Probability: {top['probability']:.4f}")
            print(f"Similarity: {top['similarity']:.4f}")
            print(f"Status: {'✓ MATCH' if top['matched'] else '✗ NO MATCH'}")
        
        print(f"\nAll Matches:")
        for i, match in enumerate(report['all_matches'][:5], 1):  # Top 5
            status = "✓" if match['matched'] else "✗"
            print(f"  {i}. {status} {match['person_name']}: {match['probability']:.4f}")
        print(f"{'='*60}\n")
    else:
        logger.error(f"Verification failed: {report['error']}")
        return 1
    
    return 0


def cmd_batch_verify(args):
    """Batch verify from directory"""
    system = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    
    if not os.path.exists(DB_PICKLE):
        logger.error("❌ No enrollment database found. Run enrollment first.")
        return 1
    
    system.load_enrollment_db(DB_PICKLE)
    
    threshold = args.threshold if args.threshold else ENROLLMENT_THRESHOLD
    results = system.batch_verify(args.dir, threshold)
    
    # Generate report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "verification_report.csv")
    system.generate_report(results, report_path)
    
    # Print summary
    matched = sum(1 for r in results if r['success'] and r['top_match'] and r['top_match']['matched'])
    total = len(results)
    logger.info(f"\n📊 Verification Summary:")
    logger.info(f"   Total tested: {total}")
    logger.info(f"   Matched: {matched}/{total} ({matched/total*100:.1f}%)")
    
    return 0


def cmd_list_enrolled(args):
    """List all enrolled people"""
    system = BiometricFaceRecognition(yolo_model=YOLO_MODEL, device=DEVICE)
    
    if not os.path.exists(DB_PICKLE):
        logger.error("❌ No enrollment database found.")
        return 1
    
    system.load_enrollment_db(DB_PICKLE)
    
    print(f"\n{'='*60}")
    print(f"ENROLLED PEOPLE ({len(system.enrollment_db)} total)")
    print(f"{'='*60}")
    for i, (person_id, name) in enumerate(system.person_names.items(), 1):
        print(f"  {i:3d}. {person_id:20s} -> {name}")
    print(f"{'='*60}\n")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Biometric Face Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  # Enroll single person (one frontal face image)
  python main.py enroll --image database/john_doe.jpg --id john_doe --name "John Doe"

  # Batch enroll all faces from directory
  python main.py batch-enroll --dir database/

  # Verify single test image
  python main.py verify --image test_images/photo.jpg

  # Batch verify all test images
  python main.py batch-verify --dir test_images/ --threshold 0.60

  # List all enrolled people
  python main.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Enroll single
    enroll_parser = subparsers.add_parser('enroll', help='Enroll single face')
    enroll_parser.add_argument('--image', required=True, help='Image path')
    enroll_parser.add_argument('--id', required=True, help='Person ID')
    enroll_parser.add_argument('--name', help='Display name')
    enroll_parser.set_defaults(func=cmd_enroll)
    
    # Batch enroll
    batch_enroll_parser = subparsers.add_parser('batch-enroll', help='Batch enroll')
    batch_enroll_parser.add_argument('--dir', default=DATABASE_DIR, help='Database directory')
    batch_enroll_parser.set_defaults(func=cmd_batch_enroll)
    
    # Verify single
    verify_parser = subparsers.add_parser('verify', help='Verify single image')
    verify_parser.add_argument('--image', required=True, help='Image path')
    verify_parser.add_argument('--threshold', type=float, help='Match threshold')
    verify_parser.set_defaults(func=cmd_verify_single)
    
    # Batch verify
    batch_verify_parser = subparsers.add_parser('batch-verify', help='Batch verify')
    batch_verify_parser.add_argument('--dir', default=TEST_DIR, help='Test directory')
    batch_verify_parser.add_argument('--threshold', type=float, help='Match threshold')
    batch_verify_parser.set_defaults(func=cmd_batch_verify)
    
    # List
    list_parser = subparsers.add_parser('list', help='List enrolled people')
    list_parser.set_defaults(func=cmd_list_enrolled)
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Execute command
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
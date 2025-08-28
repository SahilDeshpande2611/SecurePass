import os
import cv2
import pandas as pd
import shutil
import sys
from typing import List, Dict, Any
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify dependencies
try:
    from app.utils import detection_utils, ocr_utils
except ImportError as e:
    logger.error(f"Failed to import detection_utils or ocr_utils: {e}")
    sys.exit(1)

try:
    import paddle
    import paddleocr
except ImportError as e:
    logger.error(f"Missing PaddleOCR dependencies: {e}")
    sys.exit(1)

# Hard-coded paths
INPUT_DIR = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\our database"
VALID_PLATES_DIR = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES"
ORIGINAL_OUTPUT_DIR = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES\\original_images"
CROP_OUTPUT_DIR = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES\\plate_crops"
NON_RECOGNISED_DIR = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES\\non_recognised_plates"
CSV_OUTPUT_PATH = "C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES\\plate_results.csv"

def create_folders():
    """
    Create VALID_PLATES directory and its subfolders.
    """
    try:
        os.makedirs(VALID_PLATES_DIR, exist_ok=True)
        logger.info(f"Created folder: {VALID_PLATES_DIR}")
        for folder in [ORIGINAL_OUTPUT_DIR, CROP_OUTPUT_DIR, NON_RECOGNISED_DIR]:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=False)
                logger.info(f"Deleted folder: {folder}")
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Created folder: {folder}")
    except PermissionError as e:
        logger.error(f"Permission error creating folders: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating folders: {e}")
        sys.exit(1)

def test_write_permissions():
    """
    Test write permissions for all output paths by saving a dummy image.
    """
    paths = [ORIGINAL_OUTPUT_DIR, CROP_OUTPUT_DIR, NON_RECOGNISED_DIR]
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    for path in paths:
        try:
            test_file = f"{path}\\test_image.jpg"
            cv2.imwrite(test_file, dummy_image)
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.info(f"Write permission confirmed for {path}")
            else:
                logger.error(f"Failed to write test image to {path}")
                sys.exit(1)
        except PermissionError as e:
            logger.error(f"Permission denied for {path}: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error testing write permission for {path}: {e}")
            sys.exit(1)
    
    try:
        test_csv = f"{VALID_PLATES_DIR}\\test.csv"
        pd.DataFrame({"test": [1]}).to_csv(test_csv, index=False)
        if os.path.exists(test_csv):
            os.remove(test_csv)
            logger.info(f"Write permission confirmed for {VALID_PLATES_DIR}")
        else:
            logger.error(f"Failed to write test CSV to {VALID_PLATES_DIR}")
            sys.exit(1)
    except PermissionError as e:
        logger.error(f"Permission denied for {VALID_PLATES_DIR}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error testing write permission for {VALID_PLATES_DIR}: {e}")
        sys.exit(1)

def get_image_files(directory: str) -> List[str]:
    """
    Get list of image files in the specified directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        List of image file paths
    """
    logger.debug(f"Scanning directory: {directory}")
    if not os.path.exists(directory):
        logger.error(f"Input directory does not exist: {directory}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f"{directory}\\{f}" for f in os.listdir(directory)
        if os.path.isfile(f"{directory}\\{f}") and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    logger.info(f"Found {len(image_files)} image files in {directory}: {image_files}")
    return image_files

def safe_save_image(image: Any, output_path: str) -> bool:
    """
    Safely save an image to the specified path with error handling.
    
    Args:
        image: Image to save (numpy array)
        output_path: Path to save the image
        
    Returns:
        Boolean indicating success
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            logger.error(f"Cannot save image to {output_path}: Invalid image")
            return False
        success = cv2.imwrite(output_path, image)
        if success and os.path.exists(output_path):
            logger.info(f"Successfully saved image to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}: cv2.imwrite returned False or file not found")
        return success
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False

def process_image(image_path: str, seen_plates: set, debug: bool = True) -> List[Dict[str, Any]]:
    """
    Process a single image through the YOLO and OCR pipeline.
    
    Args:
        image_path: Path to the input image
        seen_plates: Set of already processed plate texts to avoid duplicates
        debug: Enable debug logging
        
    Returns:
        List of dictionaries with plate details for each detection
    """
    logger.debug(f"Processing image: {image_path}")
    results = []
    
    try:
        # Validate image path
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return [{
                "filename": os.path.basename(image_path),
                "raw_text": "",
                "plate_text": "",
                "valid": False,
                "status": "Error",
                "confidence": 0.0,
                "detection_confidence": 0.0,
                "ind_detected": False,
                "method": ""
            }]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return [{
                "filename": os.path.basename(image_path),
                "raw_text": "",
                "plate_text": "",
                "valid": False,
                "status": "Error",
                "confidence": 0.0,
                "detection_confidence": 0.0,
                "ind_detected": False,
                "method": ""
            }]
        logger.debug(f"Image loaded successfully: {image_path}")
        
        # YOLO detection
        try:
            detections = detection_utils.detect_number_plate(image_path)
            logger.debug(f"Detections: {detections}")
        except Exception as e:
            logger.error(f"Failed to detect plates in {image_path}: {e}")
            crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
            safe_save_image(image, crop_output_path)
            return [{
                "filename": os.path.basename(image_path),
                "raw_text": "",
                "plate_text": "",
                "valid": False,
                "status": "Detection Error",
                "confidence": 0.0,
                "detection_confidence": 0.0,
                "ind_detected": False,
                "method": ""
            }]
        
        # Process each detection
        if not detections:
            logger.warning(f"No plates detected in {image_path}")
            crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
            safe_save_image(image, crop_output_path)
            return [{
                "filename": os.path.basename(image_path),
                "raw_text": "",
                "plate_text": "",
                "valid": False,
                "status": "No Plate Detected",
                "confidence": 0.0,
                "detection_confidence": 0.0,
                "ind_detected": False,
                "method": ""
            }]
        
        for detection in detections:
            bbox = detection.get("bbox", None)
            detection_conf = detection.get("confidence", 0.0)
            class_name = detection.get("class", "number_plate")
            
            if not bbox or len(bbox) != 4:
                logger.warning(f"Invalid bbox in {image_path}: {bbox}")
                crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
                safe_save_image(image, crop_output_path)
                results.append({
                    "filename": os.path.basename(image_path),
                    "raw_text": "",
                    "plate_text": "",
                    "valid": False,
                    "status": "Invalid Bbox",
                    "confidence": 0.0,
                    "detection_confidence": detection_conf,
                    "ind_detected": False,
                    "method": ""
                })
                continue
            
            # Crop image
            try:
                cropped_image = ocr_utils.crop_image(image_path, bbox)
                logger.debug(f"YOLO-cropped image created for {image_path}")
            except Exception as e:
                logger.error(f"Error cropping {image_path}: {e}")
                crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
                safe_save_image(image, crop_output_path)
                results.append({
                    "filename": os.path.basename(image_path),
                    "raw_text": "",
                    "plate_text": "",
                    "valid": False,
                    "status": "Crop Error",
                    "confidence": 0.0,
                    "detection_confidence": detection_conf,
                    "ind_detected": False,
                    "method": ""
                })
                continue
            
            # OCR
            try:
                plate_details = ocr_utils.get_plate_details(cropped_image, debug=debug)
                logger.debug(f"Plate details: {plate_details}")
                
                # Check for IND in raw_text
                ind_detected = "IND" in plate_details.get("raw_text", "").upper()
                
                # Check for duplicates
                if plate_details.get("valid", False) and plate_details.get("plate_text", "") in seen_plates:
                    logger.warning(f"Duplicate plate {plate_details['plate_text']} in {image_path}")
                    results.append({
                        "filename": os.path.basename(image_path),
                        "raw_text": plate_details.get("raw_text", ""),
                        "plate_text": plate_details.get("plate_text", ""),
                        "valid": True,
                        "status": "Duplicate",
                        "confidence": plate_details.get("confidence", 0.0),
                        "detection_confidence": detection_conf,
                        "ind_detected": ind_detected,
                        "method": "original"  # Default method
                    })
                    continue
                
                # Update seen plates
                if plate_details.get("valid", False):
                    seen_plates.add(plate_details.get("plate_text", ""))
                    logger.debug(f"Added plate to seen_plates: {plate_details.get('plate_text', '')}")
                
                # Prepare result
                result = {
                    "filename": os.path.basename(image_path),
                    "raw_text": plate_details.get("raw_text", ""),
                    "plate_text": plate_details.get("plate_text", ""),
                    "valid": plate_details.get("valid", False),
                    "status": "Processed" if plate_details.get("valid", False) else "Invalid",
                    "confidence": plate_details.get("confidence", 0.0),
                    "detection_confidence": detection_conf,
                    "ind_detected": ind_detected,
                    "method": "original"  # Default method
                }
                
                # Save images
                if plate_details.get("valid", False):
                    original_output_path = f"{ORIGINAL_OUTPUT_DIR}\\{plate_details['plate_text']}.jpg"
                    safe_save_image(image, original_output_path)
                    
                    crop_output_path = f"{CROP_OUTPUT_DIR}\\{plate_details['plate_text']}_crop.jpg"
                    safe_save_image(cropped_image, crop_output_path)
                else:
                    crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
                    safe_save_image(cropped_image, crop_output_path)
                
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error in OCR for {image_path}: {e}")
                crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
                safe_save_image(cropped_image, crop_output_path)
                results.append({
                    "filename": os.path.basename(image_path),
                    "raw_text": "",
                    "plate_text": "",
                    "valid": False,
                    "status": "OCR Error",
                    "confidence": 0.0,
                    "detection_confidence": detection_conf,
                    "ind_detected": False,
                    "method": ""
                })
        
        return results
    
    except Exception as e:
        logger.error(f"Unexpected error processing {image_path}: {e}")
        crop_output_path = f"{NON_RECOGNISED_DIR}\\{os.path.basename(image_path)}_crop.jpg"
        safe_save_image(image, crop_output_path)
        return [{
            "filename": os.path.basename(image_path),
            "raw_text": "",
            "plate_text": "",
            "valid": False,
            "status": "Error",
            "confidence": 0.0,
            "detection_confidence": 0.0,
            "ind_detected": False,
            "method": ""
        }]

def main():
    """
    Main function to process images, remove duplicates, and generate CSV.
    """
    logger.info("Starting plate processing script")
    
    # Create folders
    create_folders()
    
    # Test write permissions
    test_write_permissions()
    
    # Verify input directory
    logger.debug(f"Checking input directory: {INPUT_DIR}")
    image_paths = get_image_files(INPUT_DIR)
    if not image_paths:
        logger.warning("No images found in input directory")
        test_image = f"{INPUT_DIR}\\test.jpg"
        if os.path.exists(test_image):
            logger.info(f"Testing with single image: {test_image}")
            image_paths = [test_image]
        else:
            logger.error("No test image available. Exiting.")
            return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    results = []
    seen_plates = set()
    
    for idx, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
        image_results = process_image(image_path, seen_plates, debug=True)
        for result in image_results:
            if result["status"] != "Duplicate":  # Exclude duplicates from CSV
                results.append(result)
    
    # Create DataFrame
    logger.debug("Creating DataFrame")
    df = pd.DataFrame(results, columns=[
        "filename", "raw_text", "plate_text", "valid", "status",
        "confidence", "detection_confidence", "ind_detected", "method"
    ])
    
    # Save CSV
    try:
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        logger.info(f"Saved results to {CSV_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save CSV to {CSV_OUTPUT_PATH}: {e}")
    
    # Summary
    valid_count = sum(df["valid"])
    duplicate_count = sum(1 for r in results if r["status"] == "Duplicate")
    non_recognised_count = sum(1 for r in results if not r["valid"] and r["status"] != "Duplicate")
    logger.info(f"\nSummary:")
    logger.info(f"Total images processed: {len(image_paths)}")
    logger.info(f"Valid plates detected: {valid_count}")
    logger.info(f"Duplicate plates skipped: {duplicate_count}")
    logger.info(f"Non-recognized plates: {non_recognised_count}")

if __name__ == "__main__":
    main()
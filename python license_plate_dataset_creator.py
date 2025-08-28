# improved_license_plate_dataset_creator.py

import os
import shutil
import cv2
import re
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR
from app.utils.detection_utils import load_yolo_model, detect_number_plate
from app.utils.ocr_utils import crop_image

class LicensePlateDatasetCreator:
    def __init__(self, 
                 source_dir="C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\our database",
                 output_dir="C:\\Users\\PRATIK\\SECURE_PASS_BACKEND\\data\\VALID_PLATES",
                 yolo_weights_path="./yolo_v8_custom_updated/weights/best.pt"):
        """Initialize the dataset creator for sorting license plate images."""
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.crops_dir = os.path.join(output_dir, "plate_crops")
        self.original_dir = os.path.join(output_dir, "original_images")
        
        # Create necessary directories
        for directory in [self.output_dir, self.crops_dir, self.original_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load models
        print("Loading YOLO model...")
        self.yolo_model = load_yolo_model(yolo_weights_path)
        print("Loading PaddleOCR model...")
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        
        # Initialize results tracking
        self.results_data = []
        self.total_processed = 0
        self.valid_plates = 0
        
        # Track processed plate numbers to avoid duplicates
        self.processed_plates = set()
        
        # Supported image extensions
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Patterns for extracting the correct license plate portion
        self.state_codes = [
            'AP', 'AR', 'AS', 'BR', 'CG', 'DL', 'GA', 'GJ', 'HR', 'HP',
            'JK', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL',
            'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB'
        ]
        
        # Dictionary of common OCR misrecognitions to correct
        self.char_corrections = {
            '$': 'S',
            '0': 'O',  # Only in certain contexts
            'I': '1',  # Only in certain contexts
            'B': '8',  # Only in certain contexts
            'D': '0',  # Only in certain contexts
            'Q': '0',  # Only in certain contexts
        }
    
    def get_all_images(self):
        """Get list of all image files in the source directory."""
        image_files = []
        
        # Walk through the directory
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.valid_extensions:
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        print(f"Found {len(image_files)} images in {self.source_dir}")
        return image_files
    
    def correct_characters(self, plate_text):
        """Apply character corrections based on common OCR mistakes."""
        # Replace $ with S
        if '$' in plate_text:
            plate_text = plate_text.replace('$', 'S')
        
        # Apply context-aware corrections
        # For state and district portion (first 4-5 chars), don't correct numbers to letters
        if len(plate_text) >= 4:
            prefix = plate_text[:4]
            rest = plate_text[4:]
            
            # For the registration number part (typically the last 4 digits)
            # correct letters to numbers in specific positions
            if len(rest) >= 4:
                last_part = rest[-4:]
                if not last_part.isdigit():
                    corrected_last = ''
                    for char in last_part:
                        if char == 'O':
                            corrected_last += '0'
                        elif char == 'I' or char == 'l':
                            corrected_last += '1'
                        elif char == 'B':
                            corrected_last += '8'
                        elif char == 'S' or char == '$':
                            corrected_last += '5'
                        elif char == 'Z':
                            corrected_last += '2'
                        else:
                            corrected_last += char
                    
                    # Replace only if it became more numeric
                    if sum(c.isdigit() for c in corrected_last) > sum(c.isdigit() for c in last_part):
                        plate_text = prefix + rest[:-4] + corrected_last
        
        return plate_text
    
    def clean_plate_text(self, text):
        """
        Clean license plate text by removing unwanted characters and patterns.
        Specifically handles removing 'IND' between parts of the license plate.
        """
        if not text:
            return ""
        
        # Remove dots, spaces, and convert to uppercase
        text = re.sub(r'[\s\.]', '', text).upper()
        
        # Remove phone numbers and other numeric sequences that aren't part of plates
        text = re.sub(r'\d{7,}', '', text)
        
        # Remove "IND" when it appears between state code and plate number
        # Pattern: state code (2 chars) + optional digits + IND + rest of plate
        for state_code in self.state_codes:
            if state_code in text:
                pattern = f"({state_code}\\d{{1,2}})IND(.*)"
                match = re.search(pattern, text)
                if match:
                    text = match.group(1) + match.group(2)
                    break
        
        # Handle separate "IND" detection
        text = re.sub(r'([A-Z]{2}\d{1,2})IND([A-Z0-9]{1,6})', r'\1\2', text)
        
        # Replace locations, phone numbers and other common unwanted text
        unwanted_text = [
            "SCO", "OKULNAGAR", "ONDA", "METRO", "CO0DN", "GARDEN", "PUNE", 
            "MUMBAI", "NAGPUR", "TAXI", "PRIVATE", "HIGHWAY", "POLICE"
        ]
        
        for unwanted in unwanted_text:
            text = text.replace(unwanted, "")
        
        # Apply character corrections
        text = self.correct_characters(text)
        
        return text
    
    def extract_plate_number(self, text):
        """Extract the actual license plate number from OCR text, removing extra text."""
        if not text:
            return ""
        
        # Apply initial cleaning
        text = self.clean_plate_text(text)
        
        # 1. Try to find standard Indian license plate patterns
        standard_patterns = [
            # Standard pattern: MH12AB1234
            r'([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})',
            # Alternative pattern for some states: DL5CAF4943
            r'([A-Z]{2}\d{1,2}[A-Z]{1,4}\d{1,4})',
        ]
        
        for pattern in standard_patterns:
            matches = re.search(pattern, text)
            if matches:
                return matches.group(1)
        
        # 2. If standard patterns fail, try finding state code + numbers sequence
        for state_code in self.state_codes:
            if state_code in text:
                # Find the position of the state code
                start_pos = text.find(state_code)
                
                # Extract reasonable length after state code (typical plate is 10-13 chars)
                max_length = min(13, len(text) - start_pos)
                potential_plate = text[start_pos:start_pos + max_length]
                
                # Check if this extraction has a valid structure
                if re.match(r'^[A-Z]{2}\d{1,2}[A-Z0-9]{1,8}$', potential_plate):
                    return potential_plate
        
        # 3. Last resort - look for character patterns typical of license plates
        alpha_digit_patterns = re.findall(r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}', text)
        if alpha_digit_patterns:
            return alpha_digit_patterns[0]
            
        # If all else fails, return the first 10 characters as a fallback
        if len(text) > 6:  # Only if there's enough text to potentially contain a plate
            return text[:min(10, len(text))]
            
        return text
    
    def validate_license_plate(self, text):
        """Check if text matches common license plate formats."""
        if not text or len(text) < 7:  # Too short to be valid
            return False
        
        # Basic structure check (2 letters + at least 1 digit + at least 1 letter + at least 3 digits)
        basic_pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$'
        if re.match(basic_pattern, text):
            return True
            
        # Alternative formats
        alt_patterns = [
            r'^[A-Z]{2}\d{1,2}[A-Z0-9]{5,7}$',  # Some commercial vehicles
            r'^[A-Z]{2}\d{3,5}$',               # Very old format
        ]
        
        for pattern in alt_patterns:
            if re.match(pattern, text):
                return True
        
        # State code check - should start with valid state code
        if len(text) >= 2 and text[:2] in self.state_codes:
            # Check for reasonable length and content
            alphas = sum(c.isalpha() for c in text)
            digits = sum(c.isdigit() for c in text)
            
            # Most plates have a mix of letters and numbers
            if 7 <= len(text) <= 11 and alphas >= 3 and digits >= 3:
                return True
                
        return False
    
    def extract_paddle_text(self, result):
        """Extract text from PaddleOCR result with robust parsing."""
        texts = []
        confidence = 0.0
        confidences = []
        
        try:
            if not result:
                return "", 0.0
                
            # Process PaddleOCR results
            for line in result:
                if not line:
                    continue
                    
                # Handle different result structures
                for item in line:
                    if isinstance(item, (list, tuple)) and len(item) > 1:
                        # Standard result format
                        if isinstance(item[1], tuple) and len(item[1]) > 0:
                            texts.append(item[1][0])  # Extract text
                            confidences.append(float(item[1][1]))  # Extract confidence
                        elif isinstance(item[1], str):
                            texts.append(item[1])
            
            # Calculate average confidence if available
            if confidences:
                confidence = sum(confidences) / len(confidences)
                
            # Join detected text segments
            return " ".join(texts), confidence
        except Exception as e:
            print(f"Error parsing PaddleOCR output: {e}")
            return "", 0.0
    
    def preprocess_image(self, img):
        """Apply preprocessing to enhance plate image for OCR."""
        # Convert to grayscale if not already
        if len(img.shape) > 2 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Adaptive thresholding can help with uneven lighting
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Additional processing to improve character recognition
        # Dilate slightly to connect broken character parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Remove small noise with opening operation
        opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
        
        # Return enhanced color image for PaddleOCR
        if len(img.shape) > 2 and img.shape[2] == 3:
            enhanced_color = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
            # Blend with original for better results with PaddleOCR
            enhanced_final = cv2.addWeighted(img, 0.5, enhanced_color, 0.5, 0) 
            return enhanced_final
        else:
            return opening
    
    def process_image(self, image_path, conf_threshold=0.25):
        """Process a single image and extract license plate information."""
        filename = os.path.basename(image_path)
        
        try:
            # Detect license plates using YOLO
            detections = detect_number_plate(
                image_path, model=self.yolo_model, conf_threshold=conf_threshold
            )
            
            if not detections:
                return {
                    "filename": filename,
                    "status": "No plate detected",
                    "plate_text": "",
                    "valid": False
                }
            
            best_result = None
            best_score = -1
            
            # Process each detection
            for idx, det in enumerate(detections[:3]):  # Limit to top 3 detections
                bbox = det["bbox"]
                
                # Crop the plate region
                plate_crop = crop_image(image_path, bbox)
                
                # Preprocess for better OCR
                processed_crop = self.preprocess_image(plate_crop)
                
                # Try multiple preprocessing approaches for better results
                ocr_results = []
                
                # Original crop
                ocr_result1 = self.paddle_ocr.ocr(plate_crop, cls=True)
                raw_text1, conf1 = self.extract_paddle_text(ocr_result1)
                ocr_results.append((raw_text1, conf1))
                
                # Processed crop
                ocr_result2 = self.paddle_ocr.ocr(processed_crop, cls=True)
                raw_text2, conf2 = self.extract_paddle_text(ocr_result2)
                ocr_results.append((raw_text2, conf2))
                
                # Select best OCR result based on confidence
                ocr_results.sort(key=lambda x: x[1], reverse=True)
                raw_text, confidence = ocr_results[0]
                
                # Extract just the license plate portion
                extracted_plate = self.extract_plate_number(raw_text)
                
                # Validate the extracted plate
                is_valid = self.validate_license_plate(extracted_plate)
                
                # Calculate combined score (detection confidence * OCR confidence)
                combined_score = det["confidence"] * (confidence if confidence > 0 else 0.5)
                
                # Update best result if this is better
                if (is_valid and combined_score > best_score) or (best_result is None and is_valid):
                    best_score = combined_score
                    best_result = {
                        "bbox": bbox,
                        "crop": plate_crop,
                        "raw_text": raw_text,
                        "text": extracted_plate,
                        "confidence": confidence,
                        "detection_confidence": det["confidence"],
                        "valid": is_valid
                    }
            
            if best_result and best_result["valid"]:
                return {
                    "filename": filename,
                    "status": "Processed",
                    "bbox": best_result["bbox"],
                    "crop": best_result["crop"],
                    "raw_text": best_result["raw_text"],
                    "plate_text": best_result["text"],
                    "confidence": best_result["confidence"],
                    "detection_confidence": best_result["detection_confidence"],
                    "valid": best_result["valid"]
                }
            else:
                return {
                    "filename": filename,
                    "status": "No valid plate",
                    "raw_text": best_result["raw_text"] if best_result else "",
                    "plate_text": best_result["text"] if best_result else "",
                    "valid": False
                }
        except Exception as e:
            return {
                "filename": filename,
                "status": f"Error: {str(e)}",
                "plate_text": "",
                "valid": False
            }
    
    def save_valid_plate(self, image_path, result):
        """Save valid plate details to output directory, avoiding duplicates."""
        if not result["valid"]:
            return False
        
        # Create sanitized plate text for filename - ensure all dots are removed
        plate_text = result["plate_text"].replace(" ", "").replace(".", "")
        
        # Check if this plate has already been processed (avoid duplicates)
        if plate_text in self.processed_plates:
            print(f"Skipping duplicate plate: {plate_text}")
            return False
        
        # Add to processed plates set
        self.processed_plates.add(plate_text)
        
        # Get file extension from original image
        ext = os.path.splitext(image_path)[1].lower()
        
        # Save the cropped plate - use only the license plate number as filename
        crop_filename = f"{plate_text}_crop.jpg"
        crop_path = os.path.join(self.crops_dir, crop_filename)
        cv2.imwrite(crop_path, result["crop"])
        
        # Copy the original image - use only the license plate number as filename
        orig_filename = f"{plate_text}{ext}"
        orig_path = os.path.join(self.original_dir, orig_filename)
                
        shutil.copy2(image_path, orig_path)
        
        self.valid_plates += 1
        return True
    
    def process_dataset(self):
        """Process all images in the source directory and create the dataset."""
        image_files = self.get_all_images()
        
        # Process each image with progress bar
        for img_path in tqdm(image_files, desc="Processing images"):
            self.total_processed += 1
            
            # Process the image
            result = self.process_image(img_path)
            
            # Add raw text to results if available
            raw_text = result.get("raw_text", "")
            
            # Save results
            self.results_data.append({
                "filename": os.path.basename(img_path),
                "raw_text": raw_text,
                "plate_text": result.get("plate_text", ""),
                "valid": result.get("valid", False),
                "status": result.get("status", "Unknown"),
                "confidence": result.get("confidence", 0.0) if "confidence" in result else 0.0,
                "detection_confidence": result.get("detection_confidence", 0.0) if "detection_confidence" in result else 0.0
            })
            
            # Save valid plate images (if not a duplicate)
            if result.get("valid", False) and "crop" in result:
                self.save_valid_plate(img_path, result)
            
            # Print progress every 10 images
            if self.total_processed % 10 == 0:
                print(f"\nProcessed {self.total_processed}/{len(image_files)} images. Valid plates: {self.valid_plates}")
        
        # Save results to CSV
        self.save_results()
        
        print(f"\nProcessing complete!")
        print(f"Total images processed: {self.total_processed}")
        print(f"Valid license plates found: {self.valid_plates}")
        print(f"Results saved to {self.output_dir}")
    
    def save_results(self):
        """Save processing results to CSV file."""
        results_df = pd.DataFrame(self.results_data)
        csv_path = os.path.join(self.output_dir, "plate_recognition_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Generate statistics
        if not results_df.empty:
            valid_plates = results_df[results_df["valid"] == True]
            stats = {
                "total_images": len(results_df),
                "valid_plates": len(valid_plates),
                "unique_plates": len(self.processed_plates),
                "no_plate_detected": len(results_df[results_df["status"] == "No plate detected"]),
                "error_count": len(results_df[results_df["status"].str.startswith("Error")])
            }
            
            # Save statistics
            stats_path = os.path.join(self.output_dir, "statistics.txt")
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("License Plate Dataset Statistics\n")
                f.write("==============================\n\n")
                f.write(f"Total images processed: {stats['total_images']}\n")
                f.write(f"Valid license plates found: {stats['valid_plates']}\n")
                f.write(f"Unique license plates saved: {stats['unique_plates']}\n")
                f.write(f"Images with no plate detected: {stats['no_plate_detected']}\n")
                f.write(f"Images with processing errors: {stats['error_count']}\n")
                
                if len(valid_plates) > 0:
                    f.write("\nTop detected license plates:\n")
                    top_plates = valid_plates["plate_text"].value_counts().head(10)
                    for plate, count in top_plates.items():
                        f.write(f"  {plate}: {count} occurrences\n")
                        
                    # Sample of raw text vs cleaned plate text
                    f.write("\nSample conversions (Raw OCR → Cleaned Plate):\n")
                    samples = valid_plates.sample(min(10, len(valid_plates)))
                    for _, row in samples.iterrows():
                        f.write(f"  {row['raw_text']} → {row['plate_text']}\n")

# Run the dataset creator
if __name__ == "__main__":
    creator = LicensePlateDatasetCreator()
    creator.process_dataset()
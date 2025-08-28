import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IP Webcam URL
webcam_url = "http://192.168.0.108:8080/video"

# Initialize the video capture
logger.info(f"Attempting to connect to IP webcam at {webcam_url}")
cap = cv2.VideoCapture(webcam_url)

# Check if the webcam opened successfully
if not cap.isOpened():
    logger.error("Failed to connect to the IP webcam. Please check the URL and ensure the webcam is running.")
    exit(1)

# Read a single frame
ret, frame = cap.read()
if not ret:
    logger.error("Failed to capture a frame from the IP webcam.")
    cap.release()
    exit(1)

# Save the frame as an image
output_path = "C:/Users/PRATIK/SECURE_PASS/SECURE_PASS_BACKEND/test_frame.jpg"
cv2.imwrite(output_path, frame)
logger.info(f"Successfully captured a frame and saved it to {output_path}")

# Release the capture
cap.release()
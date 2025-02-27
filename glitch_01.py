import cv2
import numpy as np
import os

# Check if image exists
image_path = './image/image.jpeg'
if not os.path.exists(image_path):
    print(f"Error: Could not find {image_path}")
    exit(1)

try:
    # Read the image in binary mode
    with open(image_path, 'rb') as f:
        img_bytes = bytearray(f.read())

    # Skip the first 2000 bytes to avoid corrupting headers
    safe_zone = 2000

    # Introduce noise in a random part of the byte array
    for _ in range(100):  # Adjust to control corruption level
        idx = np.random.randint(safe_zone, len(img_bytes))
        img_bytes[idx] = np.random.randint(256)

    output_path = './image/glitched_bytes.jpg'
    # Save the corrupted image
    with open(output_path, 'wb') as f:
        f.write(img_bytes)

    # Load and validate the corrupted image
    corrupted_img = cv2.imread(output_path)
    if corrupted_img is None:
        print("Error: Failed to load corrupted image. Try reducing corruption level.")
        exit(1)

    # Display the result
    cv2.imshow('Glitched Image', corrupted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)

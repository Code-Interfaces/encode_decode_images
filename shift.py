import cv2
import numpy as np

# Load the image
img = cv2.imread('./image/image.jpeg')

# Define shift amount
shift_x = 20  # Pixels to shift

# Shift channels
b, g, r = cv2.split(img)
b = np.roll(b, shift_x, axis=1)  # Shift blue right
r = np.roll(r, -shift_x, axis=1)  # Shift red left

# Merge channels back
glitched = cv2.merge([b, g, r])

# Save and show
cv2.imwrite('./image/glitched_shift.jpg', glitched)
cv2.imshow('RGB Shift Glitch', glitched)
cv2.waitKey(0)
cv2.destroyAllWindows()

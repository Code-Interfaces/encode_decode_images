import cv2
import numpy as np
import random

image = cv2.imread('./image/image.jpg')
glitched_image = image.copy()

num_swaps = 5000

height, width, _ = image.shape

for _ in range(num_swaps):
    x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
    x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
    glitched_image[y1, x1], glitched_image[y2,
                                           x2] = glitched_image[y2, x2], glitched_image[y1, x1]

cv2.imwrite('./image/glitched_image.jpg', glitched_image)
cv2.imshow('Glitched Image', glitched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

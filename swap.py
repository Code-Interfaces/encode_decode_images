import cv2

# Load the image
img = cv2.imread('./image/image.jpeg')

b, g, r = cv2.split(img)

image_brg = cv2.merge([g, r, b])
image_gbr = cv2.merge([r, b, g])
image_rbg = cv2.merge([g, b, r])
image_grb = cv2.merge([b, r, g])


cv2.imshow('BRG', image_brg)
cv2.imshow('GBR', image_gbr)
cv2.imshow('RBG', image_rbg)
cv2.imshow('GRB', image_grb)

cv2.waitKey(0)
cv2.destroyAllWindows()

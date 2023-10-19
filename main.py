import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the .bin file
with open('lenabin.sec', 'rb') as f:
    lena = np.fromfile(f, dtype=np.uint8)

with open('peppersbin.sec', 'rb') as f:
    pepper = np.fromfile(f, dtype=np.uint8)

# Read the lenagray, lenacolor image
lena_gray = cv.imread('lenagray.jpg', cv.IMREAD_GRAYSCALE)
lena_color = cv.imread('lena512color.jpg', cv.IMREAD_COLOR)

# Reshape the lena into a 2D array
lena = lena.reshape((256, 256))
pepper = pepper.reshape((256, 256))

# New image with half of lena and half of pepper
half_img = np.zeros((256, 256), dtype=np.uint8)
half_img[:, :128] = lena[:, :128]
half_img[:, 128:] = pepper[:, 128:]

# Reverse the image
reverse_img = np.zeros((256, 256), dtype=np.uint8)
reverse_img[:, :] = half_img[:, ::-1]

# Photo negative
negative_img = np.zeros((256, 256), dtype=np.uint8)
negative_img[:, :] = 255 - lena_gray[:, :]

# Write negative image to jpg
cv.imwrite('negative.jpg', negative_img)

# Clone the lena_color image
lena_color_clone = lena_color.copy()

# Change the color of the image
lena_color_clone[:, :, 0] = lena_color[:,:,1]
lena_color_clone[:, :, 2] = lena_color[:,:,0]
lena_color_clone[:, :, 1] = lena_color[:,:,2]

# Display clone image
cv.imshow('lena_color_clone', lena_color_clone)

# Write the clone image to jpg
cv.imwrite('lena_color_clone.jpg', lena_color_clone)

# Wait for a key press to exit
cv.waitKey(0)
cv.destroyAllWindows()


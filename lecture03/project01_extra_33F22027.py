import cv2
import numpy as np

img = cv2.imread("./image00.png", 0)  # Read out as a gray scale image
img = img.astype(float)

h, w = img.shape
filtered_img = np.zeros((h, w))

for i in range(1, h - 1):
    for j in range(1, w - 1):
        kernel = img[i - 1 : i + 2, j - 1 : j + 2].flatten()
        kernel.sort()
        filtered_img[i, j] = kernel[4]

filtered_img = (
    (filtered_img - np.min(filtered_img))
    / (np.max(filtered_img) - np.min(filtered_img))
    * 255.0
)

cv2.imshow("Input", img.astype(np.uint8))
cv2.imshow("Filtered", filtered_img.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import numpy as np
from tqdm import tqdm

blocksize = 5


def computeNCC(img1, img2):
    p1 = np.reshape(img1, -1)
    p2 = np.reshape(img2, -1)

    if np.sum(p1**2) == 0 or np.sum(p2**2) == 0:
        return 0
    # Normalized Cross Correlation
    NCC = np.sum(p1 * p2) / (np.sqrt(np.sum(p1**2)) * np.sqrt(np.sum(p2**2)))
    return NCC


def BMNCC(img1, img2):
    # return disparity map
    disparity = np.zeros(img1.shape)
    for i in tqdm(range(0, img1.shape[0])):
        for j in range(0, img1.shape[1]):
            score = computeNCC(
                img1[i : i + blocksize, j : j + blocksize],
                img2[i : i + blocksize, j : j + blocksize],
            )
            disparity[i, j] = 1 - score
    return disparity


# Read out images
img_l = cv2.imread("view0_small.png", 0)  # left image
img_r = cv2.imread("view1_small.png", 0)  # right image

disparity = BMNCC(img_l, img_r)
disparity = cv2.normalize(
    src=disparity,
    dst=None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U,
)

cv2.imshow("disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()

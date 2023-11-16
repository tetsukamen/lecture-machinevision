import numpy as np
import cv2


def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], np.float32)


def stitchImage(img1, img2, H):
    """
    Stitch img1 into img2 with a 3x3 matrix H
    """
    # compute the size of stitched image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.array([[[0, 0], [0, h1], [w1, h1], [w1, 0]]], dtype=np.float32)
    pts2 = np.array([[[0, 0], [0, h2], [w2, h2], [w2, 0]]], dtype=np.float32)
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts1 = pts1_[0, :, :]
    pts2 = pts2[0, :, :]

    # compute the size of stitched image
    pts = np.concatenate((pts1, pts2), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).ravel() - 0.5, dtype=np.float32)
    [xmax, ymax] = np.array(pts.max(axis=0).ravel() + 0.5, dtype=np.float32)

    # compute the translation matrix
    t = np.array([-xmin, -ymin], dtype=int)
    Ht = translation_matrix(t[0], t[1])  # translate
    size = (int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin)))

    # warp img1 and img2
    dst1 = cv2.warpPerspective(img1, Ht @ H, size)
    dst2 = cv2.warpPerspective(img2, Ht, size)

    # overlap dst2 on dst1
    mask = np.any(dst2 != [0, 0, 0], axis=-1)
    dst1[mask] = dst2[mask]

    return dst1


if __name__ == "__main__":
    img1 = cv2.imread("kyoto01.jpg")
    img2 = cv2.imread("kyoto02.jpg")
    img3 = cv2.imread("kyoto03.jpg")

    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    kp3, des3 = akaze.detectAndCompute(img3, None)

    # Compute matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute homography matrix
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts1, pts2)

    # stitch img1 into img2
    img4 = stitchImage(img1, img2, H)

    kp4, des4 = akaze.detectAndCompute(img4, None)

    # Compute matches
    matches = matcher.match(des3, des4)
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute homography matrix
    pts3 = np.float32([kp3[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 2)
    pts4 = np.float32([kp4[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 2)
    H, mask = cv2.findHomography(pts3, pts4)

    # stitch img3 into img4
    dst = stitchImage(img3, img4, H)

    # Visualize
    # cv2.imshow("Original1", img1)
    # cv2.imshow("Original2", img2)
    cv2.imshow("Stitched", dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

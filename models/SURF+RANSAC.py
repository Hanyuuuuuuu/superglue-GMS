import time

import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS

# def main():

img_object = cv2.imread("../img/Mikolajczyk/cars/img_7019.ppm", cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread("../img/Mikolajczyk/cars/img_7027.ppm", cv2.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    print("Could not open or find the image!")
    # return -1

# Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 800  # default: 400
start = time.time()
surf = cv2.xfeatures2d.SURF_create(800)
keypoints_object, descriptors_object = surf.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = surf.detectAndCompute(img_scene, None)

# Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor, NORM_L2 is used
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_object, descriptors_scene, 2)
print('knn_matches', len(knn_matches))

# Filter matches using the Lowe's ratio test
ratio_thresh = 0.75
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
print(good_matches)
print('good', len(good_matches))
# drawMatches函数绘制两张图像之间的匹配点
img_matches = cv2.drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# Localize the object
obj = []
scene = []
for match in good_matches:
    obj.append(keypoints_object[match.queryIdx].pt)
    scene.append(keypoints_scene[match.trainIdx].pt)

obj = np.array(obj)
scene = np.array(scene)
H, inliers = cv2.findHomography(obj, scene, cv2.RANSAC)

# Draw matches with RANSAC
good_matches_ransac = [match for i, match in enumerate(good_matches) if inliers[i]]
img_matches_ransac = cv2.drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches_ransac,
                                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
end = time.time()
print('GMS takes', end - start, 'seconds')
print('good_matches_ransac', len(good_matches_ransac))
print('img_matches_ransac', img_matches_ransac.shape)
# cv2.namedWindow("img_matches", cv2.WINDOW_NORMAL)
# cv2.imshow("img_matches", img_matches)
# cv2.imwrite("img_matches.jpg", img_matches)
# print('img_matches', img_matches.shape)

cv2.namedWindow("img_matches_ransac", cv2.WINDOW_NORMAL)
cv2.imshow("img_matches_ransac", img_matches_ransac)
cv2.imwrite("img_matches_ransac.jpg", img_matches_ransac)
cv2.waitKey(0)

# return 0
# if name == "main": main()
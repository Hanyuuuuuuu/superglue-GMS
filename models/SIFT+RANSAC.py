import time

import cv2
import numpy as np
from cv2.xfeatures2d import matchGMS

# def main():

img_object = cv2.imread("../img/Mikolajczyk/bricks/img_6592.ppm", cv2.IMREAD_GRAYSCALE)
img_scene = cv2.imread("../img/Mikolajczyk/bricks/img_6593.ppm", cv2.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    print("Could not open or find the image!")
    # return -1
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
#  使用SIFT查找关键点key points和描述符descriptors
kp1, des1 = sift.detectAndCompute(img_object, None)
kp2, des2 = sift.detectAndCompute(img_scene, None)
# print(kp1)
# kp_image1 = cv2.drawKeypoints(img_object, kp1, None)
# kp_image2 = cv2.drawKeypoints(img_scene, kp2, None)

ratio = 0.85
#  K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k = 2)

# print('knn', len(raw_matches))
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio * m2.distance:
        good_matches.append([m1])
# print(good_matches)
print('good', len(good_matches))

matches = cv2.drawMatchesKnn(img_object, kp1, img_scene, kp2, good_matches, None, flags = 2)

# Localize the object
obj = []
scene = []
good_matches = [matchs[0] for matchs in good_matches]
for match in good_matches:
    obj.append(kp1[match.queryIdx].pt)
    scene.append(kp2[match.trainIdx].pt)

obj = np.array(obj)
scene = np.array(scene)
H, inliers = cv2.findHomography(obj, scene, cv2.RANSAC)

# Draw matches with RANSAC
good_matches_ransac = [match for i, match in enumerate(good_matches) if inliers[i]]
img_matches_ransac = cv2.drawMatches(img_object, kp1, img_scene, kp2, good_matches_ransac,
                                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
end = time.time()
print('GMS takes', end - start, 'seconds')
print('good_matches_ransac', len(good_matches_ransac))
print('img_matches_ransac', img_matches_ransac.shape)
# cv2.namedWindow("img_matches", cv2.WINDOW_NORMAL)
# cv2.imshow("img_matches", img_matches)
# cv2.imwrite("img_matches.jpg", img_matches)
# print('img_matches', img_matches.shape)

# cv2.namedWindow("img_matches_ransac", cv2.WINDOW_NORMAL)
# cv2.imshow("img_matches_ransac", img_matches_ransac)
# cv2.imwrite("img_matches_ransac.jpg", img_matches_ransac)
# cv2.waitKey(0)

# return 0
# if name == "main": main()
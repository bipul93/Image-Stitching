# CSE 573 - Panorama (Image Stitching)
# Author - Bipul Kumar (bipulkum@buffalo.edu)

import sys
import cv2
import os
import argparse
import numpy as np


# # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d
# -2d-planar-homog

def find_homography(src, dest):
    # RANSAC - Get four random corresponding points
    max_inliers = 0
    homography = None
    indices = []
    for i in range(1):
        rand_indices = np.random.randint(low=0, high=len(src), size=4)
        # print(rand_indices)

        a_list = []
        for rand_i in rand_indices:
            p1 = src[rand_i][0]
            p2 = dest[rand_i][0]
            # print(p1, p2)

            p_i1 = [ -p1[0], -p1[1], -1, 0, 0, 0, p1[0]*p2[0], p1[1]*p2[0], p2[0] ]
            p_i2 = [ 0, 0, 0, -p1[0], -p1[1], -1, p1[0]*p2[1], p1[1]*p2[1], p2[1] ]

            a_list.append(p_i1)
            a_list.append(p_i2)

        u, s, v = np.linalg.svd(a_list)

        # reshape into a 3 by 3 matrix
        h = np.reshape(v[8], (3, 3))

        # normalize homography
        h = (1 / h.item(8)) * h

        inliers = 0

        for index in range(len(src)):
            if is_inlier(src[i][0], dest[i][0], h):
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            homography = h
            indices = rand_indices
    print(homography)
    return homography


def is_inlier(p1, p2, h):
    p1 = np.transpose(np.asarray([p1[0], p1[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.asarray([p2[0], p2[1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error) < 5



def main():
    dir_path = os.listdir("data/")
    image_list = sorted(list(dir_path))
    images = []
    for image_path in image_list:
        print(image_path)
        image = cv2.imread("data/" + image_path)
        image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        images.append(image)

    images.pop(0)
    images = images[::-1]

    img1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    # create SURF
    SURF = cv2.xfeatures2d.SURF_create()

    # Find keypoints and descriptors between the two images
    keypoint1, descriptor1 = SURF.detectAndCompute(img1, None)
    keypoint2, descriptor2 = SURF.detectAndCompute(img2, None)

    print(descriptor1.shape)
    print(descriptor2.shape)
    print(len(descriptor1))

    # cv2.imshow('original_image_left_keypoints', cv2.drawKeypoints(images[0], keypoint1, None))
    # cv2.imshow('original_image_right_keypoints', cv2.drawKeypoints(images[1], keypoint2, None))
    # cv2.waitKey(30000)
    # cv2.destroyAllWindows()
    # Get Corresponding points
    # ToDo: Maybe implement kd-tree based approach for faster computation

    matches = []
    for i in range(len(descriptor1)):
        distance = []
        for j in range(len(descriptor2)):
            calc_measure = cv2.norm(descriptor1[i], descriptor2[j], cv2.NORM_L2)
            distance.append([calc_measure, i, j])
        distance.sort()
        matches.append(distance[:2])
    print(matches[0])

    # Calculate good matches based on Lowe's ratio
    good_matches = []
    for m, n in matches:
        if m[0] < 0.6 * n[0]:
            good_matches.append(m)

    minimum_matches = 10
    M = None
    if len(good_matches) > minimum_matches:
        source_points = np.float32([keypoint1[int(m[1])].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([keypoint2[int(m[2])].pt for m in good_matches]).reshape(-1, 1, 2)

        M = find_homography(source_points, destination_points)

        # M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

        # print(M, source_points[0], destination_points[0])
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow("original_image_overlapping.jpg", img2)
    else:
        print("Not enough matches are found - %d/%d", (len(good_matches) / minimum_matches))

    print(images[0].shape[1] + images[1].shape[1], images[0].shape[0])
    print(images[0].shape, images[1].shape)

    # width = images[0].shape[1] + images[1].shape[1]
    # height = images[0].shape[0] + images[1].shape[0]
    #
    # dst = cv2.warpPerspective(images[0], M, (width, height))
    # dst[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

    dst = cv2.warpPerspective(images[0], M, (images[0].shape[1] + images[1].shape[1], images[0].shape[0]))
    print(dst.shape)
    dst[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]
    cv2.imshow("original_image_stitched.jpg", dst)

    def trim(frame):
        # crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        # crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        # crop top
        if not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        # crop top
        if not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    # cv2.imsave("original_image_stitched_crop.jpg", trim(dst))
    cv2.waitKey(30000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

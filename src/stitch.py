# CSE 573 - Panorama (Image Stitching)
# Author - Bipul Kumar (bipulkum@buffalo.edu)
import math
import sys
import cv2
import os
import argparse
import numpy as np


# https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d

def find_homography(src, dest):
    # RANSAC - Get four random corresponding points
    max_inliers = 0
    homography = None
    for i in range(100):
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
            if is_inlier(src[index][0], dest[index][0], h):
                inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            homography = h

    return homography


def is_inlier(p1, p2, h):
    p1 = np.transpose(np.asarray([p1[0], p1[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.asarray([p2[0], p2[1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error) < 5


def warp(img1, img2, homography):

    img1_h, img1_w = img1.shape[:2]
    img2_h, img2_w = img2.shape[:2]

    img1_pts = np.float32([[0, 0], [0, img1_h], [img1_w, img1_h], [img1_w, 0]]).reshape(-1, 1, 2)
    img2_pts = np.float32([[0, 0], [0, img2_h], [img2_w, img2_h], [img2_w, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(img2_pts, homography)

    final_dimension = np.concatenate((img1_pts, dst), axis=0)



    # Reference: https://github.com/pavanpn/Image-Stitching/blob/master/stitch_images.py

    [x_min, y_min] = np.int32(final_dimension.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(final_dimension.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = cv2.warpPerspective(img2, transform_array.dot(homography), (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:img1_h + transform_dist[1], transform_dist[0]:img1_w + transform_dist[0]] = img1

    # img2 = cv2.polylines(result_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # cv2.imshow("original_image_overlapping.jpg", result_img)

    return result_img


def stitch(images):

    img1 = cv2.GaussianBlur(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), (5, 5), 0)
    img2 = cv2.GaussianBlur(cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY), (5, 5), 0)

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

    print("Good matches:", len(good_matches))
    minimum_matches = 20
    if len(good_matches) > minimum_matches:
        source_points = np.float32([keypoint1[int(m[1])].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([keypoint2[int(m[2])].pt for m in good_matches]).reshape(-1, 1, 2)

        M = find_homography(source_points, destination_points)
        # M, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)
        # print(M)

        warped_image = warp(images[1], images[0], M)
        return warped_image

        # print(M, source_points[0], destination_points[0])

    else:
        print("Not enough matches are found - %d/%d", (len(good_matches) / minimum_matches))
        return images[1]


def main():
    arg = sys.argv
    if len(arg) < 2:
        folder = "../data/"
    else:
        folder = arg[1]
    dir_path = os.listdir(folder)
    image_list = sorted(list(dir_path))
    images = []
    for image_path in image_list:
        print(image_path)
        image = cv2.imread(folder + image_path)
        # image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        images.append(image)

    # images.pop(0)
    # images = images[::-1]

    print(len(images))

    base_image = images[0]
    for i in range(1, len(images)):
        two_image = [images[i], base_image]
        base_image = stitch(two_image)

    def trim(frame):
        if not np.sum(frame[0]):
            return trim(frame[1:])
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        if not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        if not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    def write_image(img, img_saving_path):
        """Writes an image to a given path.
        """
        if isinstance(img, list):
            img = np.asarray(img, dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if not img.dtype == np.uint8:
                assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
                img = (255 * img).astype(np.uint8)
        else:
            raise TypeError("img is neither a list nor a ndarray.")

        cv2.imwrite(img_saving_path, img)

    panorama = trim(base_image)

    write_image(panorama, os.path.join(folder, "panorama.jpg"))

    cv2.imshow("original_image_stitched_crop.jpg", panorama)

    cv2.waitKey(10000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

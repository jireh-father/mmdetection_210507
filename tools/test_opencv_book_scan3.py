import cv2  # opencv-python
import numpy as np
from skimage.filters import threshold_local  # scikit-image
import imutils
import glob
import os
from math import atan
from itertools import combinations
from sklearn.cluster import KMeans
import math

# image_dir = "E:\\resource\shoes\\triangles"
image_dir = "E:\dataset\perfitt\\real_samples"
image_files = glob.glob(os.path.join(image_dir, "*"))
output_dir = "E:\\dataset\perfitt\\real_samples_result"
os.makedirs(output_dir, exist_ok=True)


def distance(c1, c2):
    result = math.sqrt(math.pow(c1[0] - c2[0], 2) + math.pow(c1[1] - c2[1], 2))
    return result


def get_angle_between_lines(line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)


def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


for image_path in image_files:
    print(image_path)
    # read the input image
    image = cv2.imread(image_path,0)

    # clone the original image
    original_image = image.copy()

    # resize using ratio (old height to the new height)
    # ratio = image.shape[0] / 500.0
    # image = imutils.resize(image, height=500)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), image)
    image = cv2.fastNlMeansDenoising(image, h=3)
    #  change the color space to YUV
    # image = cv2.GaussianBlur(image, (3, 3), 0)

    ret, image_thr = cv2.threshold(image,
                                   180,  # threshold value
                                   255,  # maximum value assigned to pixel values exceeding the threshold
                                   cv2.THRESH_BINARY)  # threshold method type

    # cv2.imshow("threshold" + os.path.basename(image_path), image_thr)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_1_thr.jpg"), image_thr)


    ret, image_thr = cv2.threshold(image,
                                   0,  # threshold value, ignored when using cv2.THRESH_OTSU
                                   255,  # maximum value assigned to pixel values exceeding the threshold
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding type
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_2_otsu_thr.jpg"),
                image_thr)

    image_thr = cv2.adaptiveThreshold(image,
                                      255,  # maximum value assigned to pixel values exceeding the threshold
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # gaussian weighted sum of neighborhood
                                      cv2.THRESH_BINARY,  # thresholding type
                                      11,  # block size (5x5 window)
                                      5)  # constant
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_3_adp_thr.jpg"),
                image_thr)

    image = image_thr
    # image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    #
    # # grap only the Y component
    # image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
    # image_y[:, :] = image_yuv[:, :, 0]
    #
    # # blur the image to reduce high frequency noises
    # image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)

    image = cv2.bitwise_not(image)

    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_4_reverse.jpg"),
                image)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    image = cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=5
    )
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_5_morph.jpg"),
                image)


    # find edges in the image
    edges = cv2.Canny(image, 50, 200, apertureSize=5)
    # edges = cv2.Canny(image_blurred, 50, 200, apertureSize=3)

    # Step 4: Get Quadrilaterals

    # find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # draw all contours on the original image
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 1)
    # !! Attention !! Do not draw contours on the image at this point
    # I have drawn all the contours just to show below image
    # cv2.imshow("draw_contours_"+os.path.basename(image_path), image)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_6_contuors.jpg"),
                original_image)

    # to collect all the detected polygons
    polygons = []

    if not contours:
        print("skip")
        continue
    # loop over the contours
    for cnt in contours:
        # find the convex hull
        hull = cv2.convexHull(cnt)

        # compute the approx polygon and put it into polygons
        polygons.append(cv2.approxPolyDP(hull, 0.001 * cv2.arcLength(hull, True), False))

    # sort polygons in desc order of contour area
    sortedPoly = sorted(polygons, key=cv2.contourArea, reverse=True)

    # if len(sortedPoly[0]) < 3:
    #     print("skip polygon")
    #     continue

    if False:#len(sortedPoly[0]) > 4:
        polygon = np.squeeze(sortedPoly[0])

        left_top_idx = np.argmin(polygon[:, 0])
        right_bottom_idx = np.argmax(polygon[:, 1])

        right_top_corner = [original_image.shape[1] - 1, 0]

        min_dist = distance([0, 0], [original_image.shape[1] - 1, original_image.shape[0] - 1])
        min_dist_idx = -1
        for i, vertex in enumerate(polygon):
            if i in [left_top_idx, right_bottom_idx]:
                print("same idx skip")
                continue
            dist = distance(vertex, right_top_corner)
            print(dist, min_dist)
            if dist < min_dist:
                min_dist = dist
                min_dist_idx = i

        new_polygon = sortedPoly[0]
        selected_indexes = [min_dist_idx, right_bottom_idx, left_top_idx]
        new_polygon = list(new_polygon)
        for i in range(len(new_polygon) - 1, -1, -1):
            if i not in selected_indexes:
                del new_polygon[i]
    else:
        new_polygon = sortedPoly[0]
    # draw points of the intersection of only the largest polyogon with red color
    print(len(new_polygon))
    for vertex in new_polygon:
        cv2.circle(original_image, vertex[0], 3, (255, 0, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_points.jpg"),
                original_image)
    # cv2.imshow("draw_points_"+os.path.basename(image_path), image)

    # get the contours of the largest polygon in the image
    # simplified_cnt = sortedPoly[0]

cv2.waitKey(0)

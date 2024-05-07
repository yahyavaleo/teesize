import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def warp(image, distorted_points, true_width, true_height, pixeltoinch):
    true_width = true_width * pixeltoinch
    true_height = true_height * pixeltoinch

    distorted_points = np.float32(distorted_points)
    fixed_points = np.float32([[0, 0], [true_width, 0], [true_width, true_height], [0, true_height]])

    M = cv2.getPerspectiveTransform(distorted_points, fixed_points)
    return cv2.warpPerspective(image, M, (int(true_width), int(true_height)))


def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def trim(image, margin):
    height, width = image.shape[:2]
    image = image[margin : height - margin, margin : width - margin]
    return image


def detect_chessboard(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    ul_image = image[0 : (height // 2), 0 : (width // 2)]
    ur_image = image[0 : (height // 2), (width // 2) : width]
    br_image = image[(height // 2) : height, (width // 2) : width]
    bl_image = image[(height // 2) : height, 0 : (width // 2)]

    _, corners = cv2.findChessboardCorners(ul_image, (5, 5), None)
    ul = corners[12].squeeze()

    _, corners = cv2.findChessboardCorners(ur_image, (5, 5), None)
    ur = corners[12].squeeze() + [width // 2, 0]

    _, corners = cv2.findChessboardCorners(br_image, (5, 5), None)
    br = corners[12].squeeze() + [width // 2, height // 2]

    _, corners = cv2.findChessboardCorners(bl_image, (5, 5), None)
    bl = corners[12].squeeze() + [0, height // 2]

    return ul, ur, br, bl


def perspective_correction(image, true_width, true_height, pixeltoinch, margin, is_blur=False):
    points = detect_chessboard(image)
    image = warp(image, points, true_width, true_height, pixeltoinch)
    if is_blur:
        image = sharpen(image)
    image = trim(image, margin)
    return image


class Point:
    def __init__(self, coordinates):
        self.x, self.y = coordinates

    def __str__(self):
        return f"({self.x}, {self.y})"


class Line:
    def __init__(self, point1, other):
        self.point1 = point1
        if isinstance(other, Point):
            point2 = other
            self.point2 = point2
            self.A = point2.y - point1.y
            self.B = point1.x - point2.x
            self.C = self.A * point1.x + self.B * point1.y
        else:
            slope = other
            self.A = -slope
            self.B = 1
            self.C = self.A * point1.x + self.B * point1.y

    def __str__(self):
        return f"{self.A:.3f}x + {self.B:.3f}y = {self.C:.3f}"


def intersection(line1, line2):
    # Calculate the determinant
    det = line1.A * line2.B - line2.A * line1.B

    if det == 0:
        return None  # Lines are parallel and do not intersect
    else:
        # Using Cramer's Rule to solve for x and y
        x = (line2.B * line1.C - line1.B * line2.C) / det
        y = (line1.A * line2.C - line2.A * line1.C) / det
        return Point((x, y))


def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def gradient(point1, point2):
    if point1.x == point2.x:
        return None  # The gradient is undefined (vertical line)
    else:
        return (point2.y - point1.y) / (point2.x - point1.x)


def perpendicular(gradient):
    if gradient == 0:
        return None
    return -1 / gradient


def average(*args):
    return np.mean(np.array(args))


def ordered(point1, point2, point3, point4):
    if point1.y >= point3.y:
        return point1, point2, point3, point4
    return point3, point4, point1, point2


def draw_measurements(image, landmarks, lines, savepath):
    fig, ax = plt.subplots(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)

    points = np.array(landmarks)
    ax.plot(
        points[[0, 1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1, 0, 5], 0],
        points[[0, 1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1, 0, 5], 1],
        color="#9146FF",
        linewidth=2,
        linestyle="-",
        marker="s",
        markersize=4,
        markeredgecolor="black",
    )

    poly = patches.Polygon(
        points[[1, 2, 3, 4, 5, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 1], :2],
        closed=True,
        facecolor="#9146FF",
        alpha=0.3,
    )

    ax.add_patch(poly)

    for line in lines:
        x = [line.point1.x, line.point2.x]
        y = [line.point1.y, line.point2.y]
        ax.plot(x, y, color="#00ff7e", linewidth=2, linestyle="-", marker="o", markersize=4, markeredgecolor="black")

    plt.savefig(savepath)
    plt.close(fig)


def measure(landmarks, scale_factor, pixeltoinch):
    points = [Point(coords) for coords in landmarks]

    neck = distance(points[1], points[5])
    shoulder = distance(points[6], points[24])
    chest = distance(points[11], points[19])
    hem = distance(points[14], points[16])

    cuff = average(distance(points[8], points[9]), distance(points[21], points[22]))
    sleeve = average(
        (distance(points[6], points[7]) + distance(points[7], points[8])),
        (distance(points[22], points[23]) + distance(points[23], points[24])),
    )

    hline = average(gradient(points[1], points[5]), gradient(points[6], points[24]), gradient(points[14], points[16]))
    vline = perpendicular(hline)

    waist_points = ordered(points[13], points[14], points[17], points[16])
    waist_line = Line(waist_points[0], hline)
    side_seam = Line(waist_points[2], waist_points[3])
    opposite_point = intersection(waist_line, side_seam)
    waist = distance(waist_points[0], opposite_point)

    left_hem = Line(points[14], points[15])
    right_hem = Line(points[15], points[16])
    left_vline = Line(points[1], vline)
    right_vline = Line(points[5], vline)
    left_point = intersection(left_vline, left_hem)
    right_point = intersection(right_vline, right_hem)
    length = average(distance(points[1], left_point), distance(points[5], right_point))

    sizes = {
        "chest": chest,
        "waist": waist,
        "hem": hem,
        "shoulder": shoulder,
        "length": length,
        "neck": neck,
        "sleeve": sleeve,
        "cuff": cuff,
    }

    sizes = {key: value * scale_factor / pixeltoinch for key, value in sizes.items()}

    neck_line = Line(points[1], points[5])
    shoulder_line = Line(points[6], points[24])
    chest_line = Line(points[11], points[19])
    hem_line = Line(points[14], points[16])
    left_cuff_line = Line(points[8], points[9])
    right_cuff_line = Line(points[21], points[22])
    upper_left_sleeve_line = Line(points[6], points[7])
    lower_left_sleeve_line = Line(points[7], points[8])
    upper_right_sleeve_line = Line(points[23], points[24])
    lower_right_sleeve_line = Line(points[22], points[23])
    waist_line = Line(waist_points[0], opposite_point)
    left_length_line = Line(points[1], left_point)
    right_length_line = Line(points[5], right_point)

    lines = [
        neck_line,
        shoulder_line,
        chest_line,
        hem_line,
        left_cuff_line,
        right_cuff_line,
        upper_left_sleeve_line,
        lower_left_sleeve_line,
        upper_right_sleeve_line,
        lower_right_sleeve_line,
        waist_line,
        left_length_line,
        right_length_line,
    ]

    return sizes, lines

import cv2
import dlib
import numpy as np
import math


def onChange(x):
    """callback function for track bars"""
    pass


def face_detection(gray_image):
    """Detect all faces in an image"""
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(gray_image, 0)

    # testing
    # for face in faces:
    #     cv2.rectangle(image, (face.left(), face.top()),
    #                   (face.right(), face.bottom()), (0, 255, 0), 5)
    #     cv2.imshow("Face Found", image)

    return faces


def face_point(gray_image, faces):
    """find all point in a face"""
    landmark_detector = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")

    # Loop over all detected face rectangles
    faces_landmarks = []
    for i in range(0, len(faces)):
        face_area = dlib.rectangle(int(faces[i].left()),
                                   int(faces[i].top()),
                                   int(faces[i].right()),
                                   int(faces[i].bottom()))

        # For every face rectangle, run landmarkDetector
        landmarks = landmark_detector(gray_image, face_area)

        # Save all faces landmarks as a list
        faces_landmarks.append(landmarks)

        # testing
        # for landmarks in faces_landmarks:
        #     for n in range(0, 68):
        #         x = landmarks.part(n).x
        #         y = landmarks.part(n).y
        #         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        # cv2.imshow("Points Found", image)
        # cv2.imwrite("images/points.png", image)

    return faces_landmarks


def lip(faces_landmarks, mask_lips, image_hsv):
    """create mask and change the color of lips"""
    global image

    color = cv2.getTrackbarPos('hue', 'Main Window',)
    sat = cv2.getTrackbarPos('sat', 'Main Window',)

    # change color on temp colored image
    image_hsv[:, :, [0]] = color
    image_hsv[:, :, [1]] = sat + 35
    image_colored = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # find point on lip
    for landmarks in faces_landmarks:
        points = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
        arr = np.array(points)
        # if lips existed draw those point
        if points:
            cv2.fillPoly(mask_lips, [arr], (255, 255, 255))

    # testing
    # cv2.imshow("lip mask", mask_lips)

    # Mask and merge
    lips = cv2.bitwise_and(image_colored, mask_lips)
    not_lips = cv2.bitwise_and(image, cv2.bitwise_not(mask_lips))
    image = not_lips + lips


def glasses(faces_landmarks, image_glasses):
    """Put glasses on faces and change the from black_white to white."""
    black_white = cv2.getTrackbarPos('black_white', 'Main Window',)
    for landmarks in faces_landmarks:
        points = []
        lst = [0, 27, 16]
        for i in lst:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))

        glasses = image_glasses.copy()
        length = int(math.dist(points[0], points[2]))

        scale_factor = round(length/glasses.shape[1], 2)
        l = int(scale_factor * glasses.shape[1])
        h = int(scale_factor * glasses.shape[0])
        if l % 2 == 1:
            l += 1
        if h % 2 == 1:
            h += 1

        glasses = cv2.resize(glasses, (l, h), interpolation=cv2.INTER_LINEAR)

        l, w = glasses.shape[:2]

        l = l/2
        w = w/2
        y = points[1][1]
        x = points[1][0]
        if points[1][0] % 2 == 1:
            x += 1
        if points[1][1] % 2 == 1:
            y += 1

        if black_white == 0:
            roi = cv2.add(image[y - int(l): int(l) + y, x -
                                int(w): int(w) + x], glasses)
        if black_white == 1:
            roi = cv2.subtract(image[y - int(l): int(l) + y, x -
                                     int(w): int(w) + x], glasses)

        image[y - int(l): int(l) + y, x - int(w): int(w) + x] = roi


def main():
    global image

    cv2.namedWindow("Main Window")

    image = cv2.imread("girl.jpg", cv2.IMREAD_COLOR)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_glasses = cv2.imread(
        "glasses.png", cv2.IMREAD_UNCHANGED)[:, :, 3]
    image_glasses = cv2.merge((image_glasses, image_glasses, image_glasses))

    # stop program
    print("Press Q to quit")

    # allow user to change lip and glass frame color
    cv2.createTrackbar("hue", 'Main Window', 1, 180, onChange)
    cv2.createTrackbar("sat", 'Main Window', 65, 155, onChange)
    cv2.createTrackbar("black_white", 'Main Window', 0, 1, onChange)

    # Detect all faces the image
    faces = face_detection(image_gray)

    # Detect all the point on face using dlib
    landmarks = face_point(image_gray, faces)

    # loop over image for color change
    while True:

        # Change lip color
        lip(landmarks, mask, image_hsv)

        # Put glasses on
        glasses(landmarks, image_glasses)

        cv2.imshow('Main Window', image)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime, timedelta
import numpy as np
import argparse
# import i//mutils
import time
import dlib
import cv2

# import pygame

# pygame.mixer.pre_init(44100, 16, 2, 4096)
# pygame.init()
# soundObj = #pygame.mixer.Sound("F:\\Diplom\\sound.ogg")
sp = dlib.shape_predictor('E:\\Diplom\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('E:\\Diplom\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
TIMEING = 0
FLAGSLEEP = False
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream('E:\\Diplom\\test4.mp4').start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * 30 / 100)
    height = int(frame.shape[0] * 30 / 100)
    dim = (width, height)
# resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
# frame = imutils.rotate(frame, 90)
# frame = imutils.resize(frame, width=240)

    rotated = rotate_bound(resized, 270)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale frame
    rects = detector(gray, 0)
# loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = sp(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(rotated, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(rotated, [rightEyeHull], -1, (0, 255, 0), 1)
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= 50:
                cv2.putText(rotated, "SLEEP HERE", (150, 150), 0, 0.7, (0, 0, 255), 2)
        # soundObj.play()
    # otherwise, the eye aspect ratio is not below the blink
    # threshold
        else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

        # reset the eye frame counter
            COUNTER = 0
    # soundObj.stop()
    # draw the total number of blinks on the frame along with
    # the computed eye aspect ratio for the frame
        cv2.putText(rotated, "Blinks: {}".format(TOTAL), (10, 30), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(rotated, "Counter: {}".format(COUNTER), (150, 30), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(rotated, "EAR: {:.2f}".format(ear), (300, 30), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(rotated, "LEAR: {:.2f}".format(leftEAR), (10, 200), 0, 0.7, (0, 0, 255), 2)
        cv2.putText(rotated, "REAR: {:.2f}".format(rightEAR), (300, 200), 0, 0.7, (0, 0, 255), 2)

# show the frame
    cv2.imshow("Frame", rotated)
    key = cv2.waitKey(1) & 0xFF

# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
# soundObj.stop()

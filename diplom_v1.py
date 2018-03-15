from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime, timedelta
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


sp = dlib.shape_predictor('F:\\Diplom\\shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('F:\\Diplom\\dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

win1 = dlib.image_window()
win1.clear_overlay()
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
while (ret):
     ret,frame =cam.read()
     final_wide = 480
     r = float(final_wide) / frame.shape[1]
     dim = (final_wide, int(frame.shape[0] * r))
     frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     dets = detector(gray, 0)
     for k, d in enumerate(dets):
         #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
         shape = sp(frame, d)
         win1.clear_overlay()
         win1.add_overlay(d)
         win1.add_overlay(shape)
         
    
     if cv2.waitKey(10) == 0x1b: # ESC
         break
cam.release()
cv2.destroyAllWindows()

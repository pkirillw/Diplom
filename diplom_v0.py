import cv2
import numpy as np
import time
import math
#import pygame
#import os
#os.getcwd()
#pygame.mixer.pre_init(44100, 16, 2, 4096)
#pygame.init()
#soundObj = pygame.mixer.Sound("D:\\Diplom\\sound.ogg")

def rotate1(x,y,rad):
    xx = int(round(math.cos(rad)*x - math.sin(rad)*y))
    yy = int(round(math.sin(rad)*x + math.cos(rad)*y))
    return xx,yy
# 
face_cascade = cv2.CascadeClassifier('F:\\Diplom\\haarcascades\\haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('F:\\Diplom\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')

count_eyes = 0
count_face = 0;
timeing = 0
rotate = 0;
flag_update_timeing = True
flag_play_music = False
angl = 0

def detect(time,image):
    global flag_update_timeing, timeing, flag_play_music, rotate,angl
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows,cols = gray.shape
    if (angl != 0):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angl,1)
        dst = cv2.warpAffine(gray,M,(cols,rows))
    else:
        M = cv2.getRotationMatrix2D((cols/2,rows/2),0,1)
        dst = cv2.warpAffine(gray,M,(cols,rows))
    faces = face_cascade.detectMultiScale(dst, 1.1, 6, minSize=(200, 200))
    count_eyes = 0
    flagRotate = True
    while(len(faces)  == 0):
        if (rotate == 90):
            rotate = 0
            break
        
        if (flagRotate):
            angl = rotate
            flagRotate = False
        else:
            angl = 360-(rotate)
            flagRotate = True
            rotate = rotate + 5
        #print("rotate: "+str(angl))
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angl,1)
        dst = cv2.warpAffine(gray,M,(cols,rows))
        faces = face_cascade.detectMultiScale(dst, 1.1, 6, minSize=(200, 200))
        #print("Current Angle: "+str(angl)+" Count faces: " +str(len(faces)))
    cv2.imshow("dst",dst)
    count_face = 0;
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angl,1)
    image = cv2.warpAffine(image,M,(cols,rows))
    for (x,y,w,h) in faces:
        i = 0
        count_face=count_face+1
        roi_eyes = dst[y:y+h, x:x+w]
        roi_color = dst[y:y+h, x:x+w]
        
        cv2.imshow("roi_eyes",roi_eyes)
        eyes = eyes_cascade.detectMultiScale(roi_eyes, 1.1, 6, minSize=(30, 30))
        anglRadian = (angl*math.pi)/180
        for (ex,ey,ew,eh) in eyes:
            count_eyes=count_eyes+1
            endEX = x + ex 
            endEY = y + ey
            endEXEW = x + ex + ew
            endEYEH = y + ey + eh
            endEXHalfEW = x + ex + int(round((ew/2)))
            endEYHalfEH = y + ey + int(round((eh/2)))
            
            cv2.line(image,(endEX,endEYHalfEH),(endEXEW,endEYHalfEH),(255,0,255),2)
            cv2.line(image,(endEXHalfEW,endEYEH),(endEXHalfEW,endEY),(255,255,255),2)   
    M = cv2.getRotationMatrix2D((cols/2,rows/2),360-angl,1)
    image = cv2.warpAffine(image,M,(cols,rows))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,'Count face: '+str(count_face),(10,400), font, 1, (200,255,155),2)
    cv2.putText(image,'Count eyes: '+str(count_eyes),(10,450), font, 1, (200,255,155),2)
    if (count_face > 0):
        if (int(round((count_eyes/count_face))) < 2):
            if (flag_update_timeing):
                timeing = time
                flag_update_timeing = False
            if ((time - timeing) > 1):
                cv2.putText(image,'SLEEP!!!',(300,430), font, 2, (255,255,255),4)
                if(not flag_play_music):
                    #soundObj.play()
                    flag_play_music = True
        else:
            flag_update_timeing = True
            #if (flag_play_music):
                #soundObj.stop()
    return image

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cam.read()
while (ret):
     ret,frame =cam.read()
     frame = np.asarray(detect(int(round(time.time())),frame))
     cv2.imshow("features", frame)
     if cv2.waitKey(10) == 0x1b: # ESC
         break
cam.release()
cv2.destroyAllWindows()
#soundObj.stop()






















































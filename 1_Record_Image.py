# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:10:52 2019 
@author: Etermeteor

This is a file which records new training data
(fist, L, okay, palm, peace) * 20 = 100 New Data
"""
  
import cv2
# parameters
(cap_region_x_begin, cap_region_y_end) = (0.4, 0.6)  # start point/total width
(threshold, blurValue, bgSubThreshold) = (60, 41, 50)
learningRate = 0.7

# ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
cap = cv2.VideoCapture(0)
bgModel = cv2.createBackgroundSubtractorMOG2()
frame_count = 0
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('Original Frame', frame)

    fgmask = bgModel.apply(frame, learningRate = learningRate)
    res = cv2.bitwise_and(frame, frame, mask = fgmask)
    
    img = res[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imshow('thresh', thresh)
    
    frame_count += 1
    if frame_count % 100 == 0:
        count += 1
        cv2.imwrite("NewTraining/peace_%d.jpg" % count, thresh)     # save frame as JPEG file      
        print('Read a new frame: ', count)
    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
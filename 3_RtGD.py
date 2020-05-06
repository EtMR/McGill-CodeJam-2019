import cv2
import numpy as np
from keras.models import load_model

# ----- Parameters for this script -----

#model_path = 

# ----- General Settings -----
prediction = ''
action = ''
score = 0
gesture_names = {0: 'Fist', 1: 'L', 2: 'Okay', 3: 'Palm', 4: 'Peace'}
model = load_model('models/VGG_cross_validated_retrained.h5')

def predict_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}', round(max(pred_array[0]) * 100, 2), '%')
    return result, score

# ----- cv2 Parameters ------
(cap_region_x_begin, cap_region_y_end) = (0.4, 0.6)  # start point/total width
(threshold, blurValue, bgSubThreshold) = (60, 41, 50)
learningRate = 0.7

cap = cv2.VideoCapture(0)
bgModel = cv2.createBackgroundSubtractorMOG2()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    # smoothing filter
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # flip the frame horizontally
    frame = cv2.flip(frame, 1)  
    cv2.rectangle(frame, 
                  (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), 
                  (255, 0, 0), 2)
    cv2.imshow('original', frame)
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    
    # Run once background is captured
    img = res[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    cv2.imshow('thresh', thresh)
    
    frame_count += 1
    if frame_count % 100 == 0:
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_vgg(target)
        frame_count = 0
    
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
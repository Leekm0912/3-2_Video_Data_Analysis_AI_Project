import time

import cv2 as cv

from PostureDetection import PostureDetection

print(cv.__version__)
print(cv.cuda.getCudaEnabledDeviceCount())

cap = cv.VideoCapture(0)
time.sleep(2)
pd = PostureDetection(cap)
while cap.isOpened():

    frame = pd.detection()
    cv.imshow('OpenPose using OpenCV', frame)

    if cv.waitKey(1) == ord('q'):
        break

import cv2, dlib  # 이미지 표현 및 눈 좌표 얻어오기
import numpy as np  # 데이터 처리
from imutils import face_utils  # 얼굴 분석
from tensorflow import keras  # 모델 학습 및 테스트
from playsound import playsound  # 소리 재생
import threading  # 스레드 사용

cap = cv2.VideoCapture(0)
while cap.isOpened():
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    if cv2.waitKey(1) == ord('q'):
        break

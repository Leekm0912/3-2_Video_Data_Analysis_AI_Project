'''from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import datetime
import cv2 as cv
import os'''

import argparse
import tkinter as tk  # Tkinter
import threading  # 스레드 사용

from PIL import ImageTk, Image  # Pillow
import cv2 as cv  # OpenCV
import dlib  # 이미지 표현 및 눈 좌표 얻어오기
import numpy as np  # 데이터 처리
from imutils import face_utils  # 얼굴 분석
from tensorflow import keras  # 모델 학습 및 테스트
from playsound import playsound  # 소리 재생


class MainGUI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
        parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
        parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
        parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

        self.args = parser.parse_args()

        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                           ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                           ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        self.inWidth = self.args.width
        self.inHeight = self.args.height

        # 얼굴에 68개의 점을 찍음
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # 학습한 모델을 불러옴
        self.model = keras.models.load_model('models/2018_12_17_22_58_35.h5')
        self.model.summary()

        self.net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
        self.cap = cv.VideoCapture(self.args.input if self.args.input else 0)

    def camThread(self):
        cap = cv.VideoCapture(0)
        panel1 = None
        panel2 = None

        if (cap.isOpened() == False):
            print("Unable to read camera feed")

        while cap.isOpened():
            self.ret, self.frame = cap.read()
            image1 = self.video_play1()
            image2 = self.video_play2()

            if panel1 is None:
                panel1 = tk.Label(image=image1)
                panel1.image = image1
                panel1.pack(side="left")
            else:
                panel1.configure(image=image1)
                panel1.image = image1

            if panel2 is None:
                self.video_play2()
                panel2 = tk.Label(image=image2)
                panel2.image = image2
                panel2.pack(side="left")
            else:
                panel2.configure(image=image2)
                panel2.image = image2

            cv.waitKey(1)

    def video_play1(self):
        print("v1")
        frame = self.frame.copy()
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        self.net.setInput(
            cv.dnn.blobFromImage(frame, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True,
                                 crop=False))
        out = self.net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(self.BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > self.args.thr else None)

        # points에 None을 제외한 값이 2개면 사람 인식 안된거.
        if len(set(points)) <= 2:
            print("사람 없음!")
            return

        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in self.BODY_PARTS)
            assert (partTo in self.BODY_PARTS)

            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                # 코 -> 목
                if idFrom == 1 and idTo == 0:
                    # 고개를 돌렸거나, 고개를 숙였을 때.
                    if abs(points[idFrom][0] - points[idTo][0]) >= 30 \
                            or points[idFrom][1] - points[idTo][1] <= 60:
                        print("자세불량!")
                    # print(idFrom, points[idFrom], idTo, points[idTo])
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = self.net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
        # OpenCV 동영상
        return imgtk

    def video_play2(self):
        print("v2")
        frame = self.frame.copy()
        # 추출할 눈 이미지 사이즈
        IMG_SIZE = (34, 26)
        # 눈을 감은 프레임을 세줄 변수
        n_count = 0
        # 경보음 다중재생 방지 변수
        global is_playing
        is_playing = False

        # 눈을 찾아주는 함수
        def crop_eye(img, eye_points):
            x1, y1 = np.amin(eye_points, axis=0)
            x2, y2 = np.amax(eye_points, axis=0)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            w = (x2 - x1) * 1.2
            h = w * IMG_SIZE[1] / IMG_SIZE[0]

            margin_x, margin_y = w / 2, h / 2

            min_x, min_y = int(cx - margin_x), int(cy - margin_y)
            max_x, max_y = int(cx + margin_x), int(cy + margin_y)

            eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

            eye_img = self.gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

            return eye_img, eye_rect

        # main
        # img_ori = cv.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)

        img = frame
        self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = self.detector(self.gray)

        for face in faces:
            shapes = self.predictor(self.gray, face)
            shapes = face_utils.shape_to_np(shapes)

            eye_img_l, eye_rect_l = crop_eye(self.gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(self.gray, eye_points=shapes[42:48])

            eye_img_l = cv.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv.flip(eye_img_r, flipCode=1)

            eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
            eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

            pred_l = self.model.predict(eye_input_l)
            pred_r = self.model.predict(eye_input_r)

            if pred_l < 0.1 and pred_r < 0.1:
                n_count += 1
            else:
                n_count = 0
                is_playing = False
                # playsound("")

            if n_count > 100:
                cv.putText(img, "Wake up", (120, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # 스레드에서 사운드 재생
                if not is_playing:
                    is_playing = True

            # visualize
            state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
            state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

            state_l = state_l % pred_l
            state_r = state_r % pred_r

            # 색 지정
            if pred_l > 0.1:
                l_color = (255, 255, 255)
            else:
                l_color = (0, 0, 255)
            if pred_r > 0.1:
                r_color = (255, 255, 255)
            else:
                r_color = (0, 0, 255)

            cv.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(l_color),
                         thickness=2)
            cv.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(r_color),
                         thickness=2)

            cv.putText(img, state_l, tuple(eye_rect_l[0:2]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (l_color), 2)
            cv.putText(img, state_r, tuple(eye_rect_r[0:2]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (r_color), 2)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
        # OpenCV 동영상
        return imgtk


if __name__ == '__main__':
    mg = MainGUI()
    thread_img = threading.Thread(target=mg.camThread, args=())
    thread_img.daemon = True
    thread_img.start()

    root = tk.Tk()
    root.mainloop()

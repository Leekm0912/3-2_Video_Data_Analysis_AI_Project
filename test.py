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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from matplotlib import animation

import random

import EyeStatusDetection


class MainGUI:
    def __init__(self, posture_detection, eye_detection):
        self.posture_detection = posture_detection
        self.eye_detection = eye_detection
        self.root = tk.Tk()
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

        self.status_text = tk.StringVar()
        self.status_text.set("집중")
        self.status_label = tk.Label(self.root, textvariable=self.status_text, fg="black", font=("console", 30))
        self.status_label.pack(side="bottom")

    def camThread(self):
        cap = cv.VideoCapture(0)
        panel1 = None
        panel2 = None

        if (cap.isOpened() == False):
            print("Unable to read camera feed")

        while cap.isOpened():
            self.ret, self.frame = cap.read()
            image1 = self.video_play1()
            image2 = self.eye_detection.detection(self.frame)

            if panel1 is None:
                panel1 = tk.Label(image=image1)
                panel1.image = image1
                panel1.pack(side="left")
            else:
                panel1.configure(image=image1)
                panel1.image = image1

            if panel2 is None:
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
            self.status_text.set("사람 없음!")
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
                        self.status_text.set("자세불량!")
                    else:
                        self.status_text.set("집중")

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

    def animate(self, i):
        max_points = 100
        ax = plt.subplot(211, xlim=(0, 50), ylim=(0, 1024))
        self.line, = ax.plot(np.arange(max_points),
                             np.ones(max_points, dtype=np.float) * np.nan, lw=1, c='blue', ms=1)
        y = random.randint(0, 1024)
        old_y = self.line.get_ydata()
        new_y = np.r_[old_y[1:], y]
        self.line.set_ydata(new_y)
        print(new_y)
        return self.line

    def linefunc(self):
        return self.line


if __name__ == '__main__':
    mg = MainGUI(None, EyeStatusDetection.EyeStatusDetection())

    thread_img = threading.Thread(target=mg.camThread, args=())
    thread_img.daemon = True
    thread_img.start()

    '''fig = plt.figure()  # figure(도표) 생성
    canvas = FigureCanvasTkAgg(fig, master=root)  #
    canvas.get_tk_widget().pack(side="bottom")  #
    anim = animation.FuncAnimation(fig, mg.animate, init_func=mg.linefunc, frames=200, interval=50, blit=False)'''

    tk.mainloop()

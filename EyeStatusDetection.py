# 눈 깜빡임 감지 모듈
import tkinter as tk  # GUI 관련

import cv2 as cv  # 이미지 표현
import dlib  # 눈 좌표 얻어오기
import numpy as np  # 데이터 처리
from imutils import face_utils  # 얼굴 분석
from tensorflow import keras  # 모델 학습 및 테스트
from playsound import playsound  # 소리 재생
from PIL import ImageTk, Image  # Pillow

import StatusCheck


class EyeStatusDetection:
    def __init__(self):
        # 얼굴에 68개의 점을 찍음
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # 학습한 모델을 불러옴
        self.model = keras.models.load_model('models/2018_12_17_22_58_35.h5')
        self.model.summary()
        # 눈을 감은 프레임을 세줄 변수
        self.n_count = 0

    # frame을 받아 눈의 위치와 상태를 찍어서 ImageTk 객체로 변환 후 리턴
    def detection(self, frame, status_check_obj: StatusCheck.StatusCheck):
        print("v2")
        cp_frame = frame.copy()
        # 추출할 눈 이미지 사이즈
        IMG_SIZE = (34, 26)


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

        self.gray = cv.cvtColor(cp_frame, cv.COLOR_BGR2GRAY)

        faces = self.detector(self.gray)
        if len(faces) > 0:
            status_check_obj.no_eyes = False

            for face in faces:
                shapes = self.predictor(self.gray, face)
                shapes = face_utils.shape_to_np(shapes)

                eye_img_l, eye_rect_l = crop_eye(self.gray, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r = crop_eye(self.gray, eye_points=shapes[42:48])
                try:
                    eye_img_l = cv.resize(eye_img_l, dsize=IMG_SIZE)
                    eye_img_r = cv.resize(eye_img_r, dsize=IMG_SIZE)
                    eye_img_r = cv.flip(eye_img_r, flipCode=1)
                except:
                    print("오류 1")
                    return

                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

                pred_l = self.model.predict(eye_input_l)
                pred_r = self.model.predict(eye_input_r)
                print(self.n_count)
                if pred_l < 0.1 and pred_r < 0.1:
                    self.n_count += 1
                else:
                    self.n_count = 0
                    is_playing = False
                    # playsound("")

                # 졸았다고 판단하면.
                if self.n_count > 10:
                    cv.putText(cp_frame, "Wake up", (120, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    status_check_obj.close_eyes = True
                    # 스레드에서 사운드 재생
                    if not is_playing:
                        is_playing = True
                else:
                    status_check_obj.close_eyes = False

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

                cv.rectangle(cp_frame, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(l_color),
                             thickness=2)
                cv.rectangle(cp_frame, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(r_color),
                             thickness=2)

                cv.putText(cp_frame, state_l, tuple(eye_rect_l[0:2]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (l_color), 2)
                cv.putText(cp_frame, state_r, tuple(eye_rect_r[0:2]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (r_color), 2)
        else:
            status_check_obj.no_eyes = True

        img = cv.cvtColor(cp_frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
        # frame을 ImageTk 객체로 변환
        return imgtk

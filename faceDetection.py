import cv2, dlib  # 이미지 표현 및 눈 좌표 얻어오기
import numpy as np  # 데이터 처리
from imutils import face_utils  # 얼굴 분석
from tensorflow import keras  # 모델 학습 및 테스트
from playsound import playsound  # 소리 재생
import threading  # 스레드 사용

# 추출할 눈 이미지 사이즈
IMG_SIZE = (34, 26)
# 눈을 감은 프레임을 세줄 변수
n_count = 0
# 경보음 다중재생 방지 변수
global is_playing
is_playing = False

# 얼굴에 68개의 점을 찍음 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 학습한 모델을 불러옴
model = keras.models.load_model('models/2018_12_17_22_58_35.h5')
model.summary()


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

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def playSound():
    global is_playing
    t = threading.Thread(target=thread)
    # 스레드를 시작한후 join해서 동작이 멈추길 기다린 후 is_plaing을 false로 바꿔 소리 재생이 가능하도록.
    t.start()
    t.join()
    is_playing = False


def thread():
    playsound("sound.wav")


# main
# cap = cv2.VideoCapture('videos/5.mp4')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        # cv2.imshow('l', eye_img_l)
        # cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        if pred_l < 0.1 and pred_r < 0.1:
            n_count += 1
        else:
            n_count = 0
            is_playing = False
            # playsound("")

        if n_count > 100:
            cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # 스레드에서 사운드 재생
            if not is_playing:
                is_playing = True
                t = threading.Thread(target=playSound)
                t.start()

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

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(l_color), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(r_color), thickness=2)

        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (l_color), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (r_color), 2)

    # 크기 조절
    img = cv2.resize(img, dsize=(1920, 1280), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break

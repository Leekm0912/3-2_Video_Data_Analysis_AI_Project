import configparser  # ini파일 사용
import tkinter as tk  # Tkinter
import threading  # 스레드 사용

import cv2 as cv  # OpenCV

import EyeStatusDetection  # 눈 깜빡임 체크 모듈 
import PostureDetection  # 자세 체크 모듈
import StatusCheck


def camThread(cap, posture_detection_obj, eye_detection_obj, status_check_obj, status_text_obj):
    posture_detection_image_panel = None
    eye_detection_image_panel = None

    while cap.isOpened():
        ret, frame = cap.read()
        # 각각 모듈에 frame을 주고 결과를 얻어옴.
        posture_detection_image = posture_detection_obj.detection(frame, status_check_obj)
        eye_detection_image = eye_detection_obj.detection(frame, status_check_obj)

        # 패널이 None이면 pack 시켜줌
        if posture_detection_image_panel is None:
            posture_detection_image_panel = tk.Label(image=posture_detection_image)
            posture_detection_image_panel.image = posture_detection_image
            posture_detection_image_panel.pack(side="left")
        # 패널이 존재하면 업데이트 시켜줌
        else:
            posture_detection_image_panel.configure(image=posture_detection_image)
            posture_detection_image_panel.image = posture_detection_image
        # 동일
        if eye_detection_image_panel is None:
            eye_detection_image_panel = tk.Label(image=eye_detection_image)
            eye_detection_image_panel.image = eye_detection_image
            eye_detection_image_panel.pack(side="left")
        else:
            eye_detection_image_panel.configure(image=eye_detection_image)
            eye_detection_image_panel.image = eye_detection_image
        status_text_obj.set(status_check_obj.check())
        cv.waitKey(1)


if __name__ == '__main__':
    # 모듈 객체 만들기
    posture_detection = PostureDetection.PostureDetection()
    eye_detection = EyeStatusDetection.EyeStatusDetection()
    status_check = StatusCheck.StatusCheck()

    # 설정파일 불러오기
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")

    input_source = None
    # 카메라번호는 숫자. int로 변환이 필요. 변환이 안된다면 동영상 경로인 것.
    try:
        input_source = int(config["input"]["input"])
    except ValueError:
        input_source = config["input"]["input"]

    # 카메라 동작
    _cap = cv.VideoCapture(input_source)

    # GUI 부분
    root = tk.Tk()
    status_text = tk.StringVar()
    status_text.set("집중")
    status_label = tk.Label(root, textvariable=status_text, fg="black", font=("console", 30))
    status_label.pack(side="bottom")

    # thread 시작
    thread_img = threading.Thread(target=camThread, args=(_cap, posture_detection, eye_detection, status_check, status_text))
    thread_img.daemon = True
    thread_img.start()

    tk.mainloop()

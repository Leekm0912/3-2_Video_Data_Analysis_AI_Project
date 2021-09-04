import tkinter as tk  # Tkinter
import threading  # 스레드 사용

import cv2 as cv  # OpenCV

import EyeStatusDetection  # 눈 깜빡임 체크 모듈 
import PostureDetection  # 자세 체크 모듈


def camThread(posture_detection_obj, eye_detection_obj):
    cap = cv.VideoCapture(0)
    posture_detection_image_panel = None
    eye_detection_image_panel = None

    while cap.isOpened():
        ret, frame = cap.read()
        # 각각 모듈에 frame을 주고 결과를 얻어옴.
        posture_detection_image = posture_detection_obj.detection(frame)
        eye_detection_image = eye_detection_obj.detection(frame)

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

        cv.waitKey(1)


if __name__ == '__main__':
    # 모듈 객체 만들기
    posture_detection = PostureDetection.PostureDetection()
    eye_detection = EyeStatusDetection.EyeStatusDetection()

    # GUI 부분
    root = tk.Tk()
    status_text = tk.StringVar()
    status_text.set("집중")
    status_label = tk.Label(root, textvariable=status_text, fg="black", font=("console", 30))
    status_label.pack(side="bottom")

    # thread 시작
    thread_img = threading.Thread(target=camThread, args=(posture_detection, eye_detection))
    thread_img.daemon = True
    thread_img.start()

    tk.mainloop()

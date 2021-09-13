import configparser  # ini파일 사용
import tkinter as tk  # Tkinter
import threading  # 스레드 사용

import cv2 as cv  # OpenCV

import EyeStatusDetection  # 눈 깜빡임 체크 모듈 
import PostureDetection  # 자세 체크 모듈
import StatusCheck  # 상태 체크 모듈
import MakeGraph  # 그래프 모듈


def work_thread(cap, posture_detection_obj, eye_detection_obj, status_check_obj, status_text_obj, status_label_obj):
    posture_detection_image_panel = None
    eye_detection_image_panel = None

    while cap.isOpened():
        ret, frame = cap.read()
        # 각각 모듈에 frame을 주고 결과를 얻어옴.
        posture_detection_image_thread = threading.Thread(target=posture_detection_obj.detection, args=(frame, status_check_obj))
        posture_detection_image_thread.daemon = True
        posture_detection_image_thread.start()
        # posture_detection_image = posture_detection_obj.detection(frame, status_check_obj)
        eye_detection_image_thread = threading.Thread(target=eye_detection_obj.detection, args=(frame, status_check_obj))
        eye_detection_image_thread.daemon = True
        eye_detection_image_thread.start()
        # eye_detection_image = eye_detection_obj.detection(frame, status_check_obj)

        # 패널이 None이면 pack 시켜줌
        if use_postureDetection and posture_detection_obj.result_frame:
            if posture_detection_image_panel is None:
                posture_detection_image_panel = tk.Label(image=posture_detection_obj.result_frame)
                posture_detection_image_panel.image = posture_detection_obj.result_frame
                posture_detection_image_panel.pack(side="left")
            # 패널이 존재하면 업데이트 시켜줌
            else:
                posture_detection_image_panel.configure(image=posture_detection_obj.result_frame)
                posture_detection_image_panel.image = posture_detection_obj.result_frame
        # 동일
        if use_eyeDetection and eye_detection_obj.result_frame:
            if eye_detection_image_panel is None:
                eye_detection_image_panel = tk.Label(image=eye_detection_obj.result_frame)
                eye_detection_image_panel.image = eye_detection_obj.result_frame
                eye_detection_image_panel.pack(side="left")
            else:
                eye_detection_image_panel.configure(image=eye_detection_obj.result_frame)
                eye_detection_image_panel.image = eye_detection_obj.result_frame
        text = StatusCheck.StatusCheck.check_str[status_check_obj.check()]

        if text != "집중":
            status_label_obj["fg"] = "red"
        else:
            status_label_obj["fg"] = "black"
        status_text_obj.set(text)
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

    use_eyeDetection = int(config["GUI"]["use_eyeDetection"])
    use_postureDetection = int(config["GUI"]["use_postureDetection"])
    use_graph = int(config["GUI"]["use_graph"])

    # 카메라 동작
    _cap = cv.VideoCapture(input_source)

    # GUI 부분
    root = tk.Tk()
    status_text = tk.StringVar()
    status_text.set("집중")
    status_label = tk.Label(root, textvariable=status_text, fg="black", font=("맑은고딕", 30))
    status_label.pack(side="bottom")

    # 그래프
    if use_graph:
        mg = MakeGraph.MakeGraph(root, status_check)
        graph = threading.Thread(target=mg.start_graph, args=())
        graph.daemon = True
        graph.start()

    # thread 시작
    thread_img = threading.Thread(target=work_thread, args=(_cap, posture_detection, eye_detection, status_check, status_text, status_label))
    thread_img.daemon = True
    thread_img.start()

    tk.mainloop()

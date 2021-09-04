# 자세 감지 모듈
import configparser

import cv2 as cv
from PIL import ImageTk, Image  # Pillow

import StatusCheck


class PostureDetection:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini", encoding="utf-8")

        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                           ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                           ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        self.inWidth = int(self.config["posture"]["width"])
        self.inHeight = int(self.config["posture"]["height"])

        self.net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    def detection(self, frame, status_check_obj: StatusCheck.StatusCheck):
        print("v1")
        nose_detected = False
        cp_frame = frame.copy()
        frameWidth = cp_frame.shape[1]
        frameHeight = cp_frame.shape[0]

        self.net.setInput(
            cv.dnn.blobFromImage(cp_frame, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True,
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
            points.append((int(x), int(y)) if conf > float(self.config["posture"]["thr"]) else None)

        # points에 None을 제외한 값이 2개면 사람 인식 안된거.
        if len(set(points)) <= 2:
            print("몸 인식 X")
            status_check_obj.no_body = True
        else:
            status_check_obj.no_body = False

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
                    if abs(points[idFrom][0] - points[idTo][0]) >= 30:
                        print("자세불량!(좌우)")
                        status_check_obj.turn_head_LR = True
                    elif abs(points[idFrom][1] - points[idTo][1]) <= 60:
                        print("자세불량!(상하)")
                        status_check_obj.turn_head_UD = True
                    else:
                        status_check_obj.turn_head_LR = False
                        status_check_obj.turn_head_UD = False

                    # print(idFrom, points[idFrom], idTo, points[idTo])
                cv.line(cp_frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(cp_frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(cp_frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

        t, _ = self.net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(cp_frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        img = cv.cvtColor(cp_frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=img)  # ImageTk 객체로 변환
        # OpenCV 동영상
        return imgtk

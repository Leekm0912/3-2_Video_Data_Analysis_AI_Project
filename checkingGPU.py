from tensorflow.python.client import device_lib
import cv2 as cv
if __name__ == '__main__':
    print(device_lib.list_local_devices())
    print(cv.__version__)
    print(cv.cuda.getCudaEnabledDeviceCount())
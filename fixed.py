import numpy as np
import cv2

def left_fixed_images(frame):

    # 저장된 카메라 캘리브레이션 파라미터
    dist = np.array([-0.40892842,  0.2169821,   0.00179793,  0.00056867, -0.09654193])
    mtx = np.array([[467.34949482, 0, 321.18797431],
                    [0, 467.15756567, 218.55614416],
                    [0, 0, 1]])

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 왜곡 보정
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # ROI 영역 잘라내기
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y + h_roi, x:x + w_roi]

    return dst_cropped

def right_fixed_images(frame):
    # 저장된 카메라 캘리브레이션 파라미터
    dist = np.array([-0.39485907,  0.18627685,  0.00142744, -0.00115882, -0.04600605])
    mtx = np.array([[455.52658117, 0, 317.78657271],
                    [0, 457.2466953, 245.92974825],
                    [0, 0, 1]])

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 왜곡 보정
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # ROI 영역 잘라내기
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y + h_roi, x:x + w_roi]

    return dst_cropped


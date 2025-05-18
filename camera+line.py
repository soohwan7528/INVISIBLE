from Crosswalk import process_frame
from fixed import left_fixed_images, right_fixed_images
import cv2
import Stereo_algorithm as algorithm
from remove_plane import stereo

# 웹캠 캡처 객체 생성
cap1 = cv2.VideoCapture('video1_20250516_165859.avi')
cap2 = cv2.VideoCapture('video2_20250516_165859.avi')

# 창 생성 및 크기 조절 가능 설정
cv2.namedWindow('Crosswalk Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Crosswalk Detection', 600,400)  # 너비 480, 높이 320

cv2.namedWindow('Crosswalk Detection1', cv2.WINDOW_NORMAL) # 두 번째 창 생성
cv2.resizeWindow('Crosswalk Detection1', 600, 400) # 두 번째 창 크기 설정

while True:
    # 프레임 읽기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1:
        # 원본 이미지를 fixed.py로 보내 보정된 이미지 받아오기
        dst_cropped1 = left_fixed_images(frame1)
        dst_cropped2 = right_fixed_images(frame2)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        depth, img_with_boxes = stereo(gray1,gray2)
        print(depth)

        # 보정된 프레임을 Crosswalk.py로 보내 처리
        processed_frame, roi_lines, message = process_frame(dst_cropped1)

        # 처리 결과 보여주기
        cv2.imshow('Crosswalk Detection', img_with_boxes)  # 변경 없음
        cv2.imshow('Crosswalk Detection1', frame2)

    else:
        print("프레임을 읽을 수 없습니다. 웹캠 연결을 확인하거나 영상 파일 경로를 확인해주세요.")
        break

    # 'q' 키를 누르면 종료
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 웹캠 캡처 객체 해제 및 창 닫기
cap1.release()
cap2.release()
cv2.destroyAllWindows()
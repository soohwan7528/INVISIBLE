import cv2
import numpy as np

# 비디오 파일 경로 (더 이상 직접 사용하지 않음)
# video_file = "video/crossline_2.mp4"

# --- 프레임 처리 함수  ---
def process_frame(frame, roi_ratio_y_start=0.5, roi_ratio_height=0.5):
    # (기존 코드와 동일)
    if frame is None:
        return None, [], None

    h, w = frame.shape[:2]

    roi_y_start = int(h * roi_ratio_y_start)
    roi_y_end = int(roi_y_start + h * roi_ratio_height)
    roi_x_start = 0
    roi_x_end = w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    edge_gray = cv2.Canny(blur, 50, 150)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([179, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 구조화 커널 생성
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))    

    # 팽창(Dilation) 적용
    mask = cv2.dilate(frame, k, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges_hsv = cv2.Canny(mask, 50, 150)

    edges = cv2.bitwise_and(edge_gray, edges_hsv)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=190, maxLineGap=20)

    detected_lines_in_roi = []
    output_frame = frame.copy()
    message = None
    first_message_generated = False

    cv2.rectangle(output_frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if roi_y_start <= center_y <= roi_y_end and roi_x_start <= center_x <= roi_x_end:
                cv2.line(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                detected_lines_in_roi.append(line[0])

                if not first_message_generated:
                    if x2 - x1 == 0:
                        slope = float('inf')
                    else:
                        slope = (y2 - y1) / (x2 - x1)

                    if slope >= 0.3:
                        message = "turn your body right"
                        first_message_generated = True
                    elif slope <= -0.3:
                        message = "turn your body left"
                        first_message_generated = True

    if message:
        cv2.putText(output_frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    return output_frame, detected_lines_in_roi, message

# --- 비디오 처리 메인 함수 ---
def process_video(frame):  # 이제 frame 하나만 받음
    processed_frame, roi_lines, current_message = process_frame(frame, roi_ratio_y_start=0.5, roi_ratio_height=0.5)

    # 처리된 프레임 표시 (메인 루프에서 처리하도록 변경)
    return processed_frame, roi_lines, current_message
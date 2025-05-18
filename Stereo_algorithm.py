import cv2
import numpy as np

def stereo(imgL, imgR):
    # --- Disparity 계산 --- 테스트 수정
    # numDisparities와 blockSize는 이미지에 맞게 조절하는 것이 좋습니다.
    # 가까운 물체의 최대 disparity를 포함하도록 numDisparities 설정이 중요할 수 있습니다.
    stereo = cv2.StereoSGBM_create(blockSize = 7, numDisparities = 96, speckleWindowSize= 100, speckleRange = 100) # 흑백만 인식함
    disparity = stereo.compute(imgL, imgR)
    disparity = np.where(disparity== -16, 0, disparity)

    # Normalized (0-255) 값 기준
    min_disp_thresh_close = 350 # 예시 값 (더 가까운 것만 원하면 값을 높임)
    max_disp_thresh_close = int(np.max(disparity)) # 최대값은 255로 설정하여 가장 가까운 영역까지 포함
    # 2. 마스크 생성 (해당 범위 내 픽셀은 255, 나머지는 0)
    mask_close = cv2.inRange(disparity, min_disp_thresh_close, max_disp_thresh_close)

    # (선택 사항) 노이즈 제거를 위한 Morphological 연산
    kernel = np.ones((3,3),np.uint8) # 작은 커널 사용 고려
    mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)

    # 3. 컨투어 찾기
    contours_close, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 원본 이미지에 컬러로 박스를 그리기 위해 BGR로 변환
    imgL_color_close = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)

    min_contour_area_close = 700

    # 모든 바운딩 박스 정보 저장
    all_boxes = []
    for contour in contours_close:
        area = cv2.contourArea(contour)
        if area > min_contour_area_close:
            x, y, w, h = cv2.boundingRect(contour)
            all_boxes.append((x, y, w, h))

    # 각 바운딩 박스에 대해 시차 계산
    bounding_Box = {'x_location':[], 'y_location':[], 'disparity':[], 'box_count':[], 'object':[]}
    box_index = 1

    for i, (x, y, w, h) in enumerate(all_boxes):
        # 현재 바운딩 박스에 대한 마스크 생성
        current_mask = np.zeros_like(disparity, dtype=np.uint8) # disparity 배열과 길이가 같은 0으로 찬 배열을 만든다.
        current_mask[y:y+h, x:x+w] = 1

        # 다른 바운딩 박스들과 겹치는 영역에 대한 마스크 생성
        overlap_mask = np.zeros_like(disparity, dtype=np.uint8) # disparity 배열과 길이가 같은 0으로 찬 배열을 만든다.
        for j, (x2, y2, w2, h2) in enumerate(all_boxes):
            if i != j:  # 자기 자신은 제외
                # 겹치는 영역 계산
                x_overlap = max(0, min(x+w, x2+w2) - max(x, x2))
                y_overlap = max(0, min(y+h, y2+h2) - max(y, y2))

                if x_overlap > 0 and y_overlap > 0:  # 겹치는 부분이 있으면
                    x_start = max(x, x2)
                    y_start = max(y, y2)
                    overlap_mask[y_start:y_start+y_overlap, x_start:x_start+x_overlap] = 1

        # 겹치지 않는 영역만 마스크로 선택
        exclusive_mask = current_mask.astype(bool) & ~overlap_mask.astype(bool)

        # 해당 영역 내에서 시차 계산 (기존 조건도 적용)
        roi = disparity[y:y+h, x:x+w]
        roi_norm = disparity[y:y+h, x:x+w]

        # 바운딩 박스에서 겹치지 않는 부분 마스크
        exclusive_roi_mask = exclusive_mask[y:y+h, x:x+w]

        # 시차 조건과 겹치지 않는 영역 조건을 모두 적용
        combined_mask = exclusive_roi_mask & (roi_norm >= 180) & (roi_norm <= 255) & (roi > 0)
        valid_disp = roi[combined_mask]

        # 바운딩 박스 그리기 (파란색)
        cv2.rectangle(imgL_color_close, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if valid_disp.size > 0:
            mean_disp = np.mean(valid_disp)
            #min_disp = np.min(valid_disp) # 고려 대상
            #max_disp = np.max(valid_disp) # 고려 대상

            z = (8 * 456.38663823499996 / mean_disp) * 4

            bounding_Box['x_location'].append([x, x + w])
            bounding_Box['y_location'].append([y, y + h])
            bounding_Box['disparity'].append(round(float(z),2)) # round(float(mean_disp), 2)
            bounding_Box['box_count'].append(box_index)

        box_index += 1

    return bounding_Box, imgL_color_close
#bounding_Box, imgL_color_close

# --- 결과 이미지 표시 ---
''''''''''''''''
from matplotlib import pyplot as plt
# 마스크 이미지 표시 (디버깅용)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.imshow(mask_close, 'gray')
plt.title(f'Mask for Close Objects ({min_disp_thresh_close}-{max_disp_thresh_close})')

# Disparity Map (정규화) 표시
plt.subplot(1, 3, 2)
plt.imshow(disparity_norm, 'gray')
plt.title('Normalized Disparity Map')

# 바운딩 박스가 그려진 원본 이미지 표시
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(imgL_color_close, cv2.COLOR_BGR2RGB))
plt.title('Image with Bounding Boxes for Close Objects')

plt.tight_layout()
plt.show()
'''

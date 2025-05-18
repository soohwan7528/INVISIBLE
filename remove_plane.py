import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from matplotlib import pyplot as plt

def remove_plane_mask(disparity, mask_close, min_points=5000):
    """
    시차맵에서 평면(바닥/벽) 영역을 RANSAC으로 검출해 마스크로 제거
    """
    h, w = disparity.shape
    ys, xs = np.where(mask_close > 0)
    ds = disparity[ys, xs].astype(np.float32)
    if len(ds) < min_points:
        return mask_close
    X = np.stack([xs, ys], axis=1)
    y = ds
    ransac = RANSACRegressor(residual_threshold=2.0, min_samples=100)
    ransac.fit(X, y)
    y_pred = ransac.predict(X)
    inlier_mask = np.abs(y - y_pred) < 2.0
    plane_mask = np.zeros_like(mask_close)
    plane_mask[ys[inlier_mask], xs[inlier_mask]] = 255
    mask_no_plane = cv2.bitwise_and(mask_close, cv2.bitwise_not(plane_mask))
    return mask_no_plane

def stereo(imgL, imgR):
    # --- Disparity 계산 ---
    stereo = cv2.StereoSGBM_create(
        blockSize=7, numDisparities=96, speckleWindowSize=100, speckleRange=100
    )
    disparity = stereo.compute(imgL, imgR)
    disparity = np.where(disparity == -16, 0, disparity)

    # Normalized (0-255) 값 기준
    min_disp_thresh_close = 350
    max_disp_thresh_close = int(np.max(disparity))
    mask_close = cv2.inRange(disparity, min_disp_thresh_close, max_disp_thresh_close)
    kernel = np.ones((3, 3), np.uint8)
    mask_close = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel)

    # --- 평면(바닥/벽) 제거 ---
    mask_no_plane = remove_plane_mask(disparity, mask_close)

    # --- 컨투어 찾기 ---
    contours_close, _ = cv2.findContours(mask_no_plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 원본 이미지에 컬러로 박스 그리기 ---
    imgL_color_close = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    min_contour_area_close = 700
    all_boxes = []
    for contour in contours_close:
        area = cv2.contourArea(contour)
        if area > min_contour_area_close:
            x, y, w, h = cv2.boundingRect(contour)
            all_boxes.append((x, y, w, h))

    # --- 각 바운딩 박스에 대해 시차 계산 ---
    bounding_Box = {'x_location': [], 'y_location': [], 'disparity': [], 'box_count': [], 'object': []}
    box_index = 1
    for i, (x, y, w, h) in enumerate(all_boxes):
        current_mask = np.zeros_like(disparity, dtype=np.uint8)
        current_mask[y:y+h, x:x+w] = 1

        overlap_mask = np.zeros_like(disparity, dtype=np.uint8)
        for j, (x2, y2, w2, h2) in enumerate(all_boxes):
            if i != j:
                x_overlap = max(0, min(x+w, x2+w2) - max(x, x2))
                y_overlap = max(0, min(y+h, y2+h2) - max(y, y2))
                if x_overlap > 0 and y_overlap > 0:
                    x_start = max(x, x2)
                    y_start = max(y, y2)
                    overlap_mask[y_start:y_start+y_overlap, x_start:x_start+x_overlap] = 1

        exclusive_mask = current_mask.astype(bool) & ~overlap_mask.astype(bool)
        roi = disparity[y:y+h, x:x+w]
        roi_norm = disparity[y:y+h, x:x+w]
        exclusive_roi_mask = exclusive_mask[y:y+h, x:x+w]
        combined_mask = exclusive_roi_mask & (roi_norm >= 180) & (roi_norm <= 255) & (roi > 0)
        valid_disp = roi[combined_mask]

        cv2.rectangle(imgL_color_close, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if valid_disp.size > 0:
            mean_disp = np.mean(valid_disp)
            z = (8 * 456.38663823499996 / mean_disp) * 4
            bounding_Box['x_location'].append([x, x + w])
            bounding_Box['y_location'].append([y, y + h])
            bounding_Box['disparity'].append(round(float(z), 2))
            bounding_Box['box_count'].append(box_index)
            box_index += 1

    return bounding_Box, imgL_color_close

# ----------------- 결과 이미지 표시 예시 -----------------

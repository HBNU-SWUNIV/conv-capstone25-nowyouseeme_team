import pyrealsense2 as rs
import cv2
import torch
import numpy as np

# YOLOv5 모델 로드 (학습된 모델 경로)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/lim/yolov5/runs/train/hope5/weights/best.pt')  # 절대 경로로 수정

# 모델 추론 설정
model.conf = 0.5         # confidence threshold
model.iou = 0.45

# RealSense 카메라 스트리밍 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 640x480 해상도, 30 FPS
pipeline.start(config)

# 실시간 객체 탐지 루프
while True:
    # RealSense 카메라에서 이미지 가져오기
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    
    # OpenCV로 이미지를 변환
    color_image = np.asanyarray(color_frame.get_data())
    
    # YOLOv5 모델로 추론
    results = model(color_image)
    
    # 탐지된 객체가 포함된 이미지 표시
    img = results.render()[0]  # 탐지된 이미지를 얻어옵니다.

    # OpenCV 창에 실시간 비디오로 표시
    cv2.imshow('RealSense YOLOv5 Detection', img)
    
    # ESC 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:  # 27은 ESC 키의 ASCII 코드입니다
        print("프로그램 종료 중...")
        break  # ESC 키를 눌렀을 때 루프 종료

# 스트리밍 종료
pipeline.stop()
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다

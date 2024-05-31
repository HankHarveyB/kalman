import torch
import numpy as np

import cv2
import numpy as np

import utils

import os

from detector import Detector

def save_detections_to_file(detections, frame_number, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'frame_{frame_number:04d}.txt'), 'w') as f:
        for detection in detections:
            print("Detection:", detection)  # 输出检测结果以调试
            x1, y1, x2, y2 = detection[:4]  # 使用切片操作以确保只取前四个值
            f.write(f'0 {x1} {y1} {x2} {y2}\n')

def process_video(video_path, output_dir, detector):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        person_detections = [detection for detection in detections if detection[4] == 'person']
        save_detections_to_file(person_detections, frame_number, output_dir)
        frame_number += 1
    cap.release()

if __name__ == '__main__':
    video_path = './data/kalman.mp4'
    output_dir = './data/Test_labels'

    
    detector = Detector()
    process_video(video_path, output_dir, detector)
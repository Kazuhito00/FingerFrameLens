#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[summary]
  FingerFrame Lens
[description]
  -
"""

import argparse
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import CvFpsCalc
from gui.app_gui import AppGui


def get_args():
    """
    [summary]
        引数解析
    Parameters
    ----------
    None
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model", default='model/EfficientDetD0/saved_model')
    parser.add_argument("--score_th", type=float, default=0.7)

    args = parser.parse_args()

    return args


def run_inference_single_image(image, inference_func):
    """
    [summary]
        物体検出推論(1枚)
    Parameters
    ----------
    image : image
        推論対象の画像
    inference_func : func
        推論用関数
    None
    """

    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output


def main():
    """
    [summary]
        main()
    Parameters
    ----------
    None
    """
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    fps = args.fps

    model_path = args.model
    score_th = args.score_th

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    DEFAULT_FUNCTION_KEY = 'serving_default'
    loaded_model = tf.saved_model.load(model_path)
    inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

    # GUI準備 #################################################################
    app_gui = AppGui(window_name='FingerFrameLens')

    # 初期設定
    app_gui.set_score_threshold(score_th)

    # FPS計測準備 ##############################################################
    cvFpsCalc = CvFpsCalc(buffer_len=3)

    while True:
        start_time = time.time()

        # GUI設定取得 #########################################################
        score_th = app_gui.get_score_threshold()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 検出実施 #############################################################
        frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
        image_np_expanded = np.expand_dims(frame, axis=0)

        output = run_inference_single_image(image_np_expanded, inference_func)

        num_detections = output['num_detections']
        for i in range(num_detections):
            score = output['detection_scores'][i]
            bbox = output['detection_boxes'][i]
            # class_id = output['detection_classes'][i].astype(np.int)

            if score < score_th:
                continue

            # 検出結果可視化 ###################################################
            risize_ratio = 0.1
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            cv.putText(debug_image, '{:.3f}'.format(score), (x1, y1 - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # GUI描画更新 ##########################################################
        fps_result = cvFpsCalc.get()
        app_gui.update(
            fps_result,
            debug_image,
        )
        app_gui.show()

        # キー入力(ESC:プログラム終了) #########################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

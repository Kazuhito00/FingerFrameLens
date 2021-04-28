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
    parser.add_argument("--smaller_ratio", type=float, default=0.22)

    args = parser.parse_args()

    return args


def run_od_inference(inference_func, image):
    """
    [summary]
        物体検出推論(1枚)
    Parameters
    ----------
    inference_func : func
        推論用関数
    image : image
        推論対象の画像
    None
    """
    image = image[:, :, [2, 1, 0]]  # BGR2RGB
    image = np.expand_dims(image, axis=0)
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output


def calc_od_bbox(detection_result, score_th, smaller_ratio, frame_width,
                 frame_height):
    """
    [summary]
        物体検出結果からバウンディングボックスを算出
    Parameters
    ----------
    detection_result : dict
        物体検出結果
    score_th : float
        物体検出スコア閾値
    smaller_ratio : float
        縮小割合
    frame_width : int
        画像幅
    frame_height : int
        画像高さ
    None
    """
    x1, y1, x2, y2 = None, None, None, None

    num_detections = detection_result['num_detections']
    for i in range(num_detections):
        score = detection_result['detection_scores'][i]
        bbox = detection_result['detection_boxes'][i]

        if score < score_th:
            continue

        # 検出結果可視化 ###################################################
        x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
        x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

        risize_ratio = smaller_ratio
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        x1 = x1 + int(bbox_width * risize_ratio)
        y1 = y1 + int(bbox_height * risize_ratio)
        x2 = x2 - int(bbox_width * risize_ratio)
        y2 = y2 - int(bbox_height * risize_ratio)

        break  # 有効なバウンディングボックスの1つ目を利用

    return x1, y1, x2, y2


def run_classify(model, image):
    """
    [summary]
        画像クラス分類
    Parameters
    ----------
    model : model
        クラス分類用モデル
    image : image
        推論対象の画像
    None
    """
    inp = cv.resize(image, (224, 224))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    inp = np.expand_dims(inp, axis=0)
    tensor = tf.convert_to_tensor(inp)
    tensor = tf.keras.applications.efficientnet.preprocess_input(tensor)

    classifications = model.predict(tensor)

    classifications = tf.keras.applications.efficientnet.decode_predictions(
        classifications,
        top=5,
    )
    classifications = np.squeeze(classifications)
    return classifications


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
    smaller_ratio = args.smaller_ratio

    # GUI準備 #################################################################
    app_gui = AppGui(window_name='FingerFrameLens')
    # 初期設定
    app_gui.set_score_threshold(score_th)

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    # EfficientDet-D0
    DEFAULT_FUNCTION_KEY = 'serving_default'
    effdet_model = tf.saved_model.load(model_path)
    inference_func = effdet_model.signatures[DEFAULT_FUNCTION_KEY]

    # EfficientNet-B0
    effnet_model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
    tensor = tf.convert_to_tensor(np.zeros((1, 224, 224, 3), np.uint8))
    effnet_model.predict(tensor)
    effnet_model.make_predict_function()

    # FPS計測準備 ##############################################################
    cvFpsCalc = CvFpsCalc(buffer_len=3)

    cropping_image = None
    classifications = None

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

        # 物体検出実施 #########################################################
        detections = run_od_inference(inference_func, frame)
        x1, y1, x2, y2 = calc_od_bbox(
            detections,
            score_th,
            smaller_ratio,
            frame_width,
            frame_height,
        )

        # cv.putText(debug_image, '{:.3f}'.format(score), (x1, y1 - 10),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
        #            cv.LINE_AA)
        # cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # クラス分類実施 #######################################################
        if x1 is not None and y1 is not None and \
           x2 is not None and y2 is not None:
            cropping_image = copy.deepcopy(frame[y1:y2, x1:x2])
            classifications = run_classify(effnet_model, cropping_image)

        # GUI描画更新 ##########################################################
        fps_result = cvFpsCalc.get()
        app_gui.update(
            fps_result,
            debug_image,
            cropping_image,
            classifications,
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

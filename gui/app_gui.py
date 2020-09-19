#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import cv2 as cv
import gui.cvui as cvui


class AppGui:
    """
    [summary]
    アプリケーションウィンドウクラス
    [description]
    -
    """
    _window_position = [0, 0]

    _score_threshold = [1]

    _cvuiframe = None

    _window_name = ''

    def __init__(self, window_name='DEBUG', window_position=None):
        self._score_threshold[0] = 0.0

        self._cvuiframe = np.zeros((456, 806 + 200, 3), np.uint8)
        self._cvuiframe[:] = (49, 52, 49)

        loading_image = cv.imread('gui/image/loading.png')
        loading_image = cv.resize(loading_image, (806 + 200, 456))
        cvui.image(self._cvuiframe, 0, 0, loading_image)

        self._window_name = window_name
        cvui.init(self._window_name)
        # cv.setWindowProperty(self._window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        self._window_position = window_position

        cvui.imshow(self._window_name, self._cvuiframe)
        cv.waitKey(1)

    def update(self, fps, frame, cropping_frame, classifications):
        """
        [summary]
          描画内容更新
        """
        self._cvuiframe[:] = (49, 52, 49)

        # 画像：撮影映像
        display_frame = copy.deepcopy(frame)
        display_frame = cv.resize(display_frame, (800, 450))
        cvui.image(self._cvuiframe, 3, 3, display_frame)

        # 文字列：FPS
        cvui.printf(self._cvuiframe, 800 + 15, 15, 0.4, 0xFFFFFF,
                    'FPS : ' + str(fps))

        # 画像：切り抜き画像
        cvui.rect(self._cvuiframe, 800 + 15 - 1, 40 - 1, 181, 181, 0xFFFFFF)
        if cropping_frame is not None:
            display_cropping_frame = copy.deepcopy(cropping_frame)
            display_cropping_frame = cv.resize(display_cropping_frame,
                                               (180, 180))
            cvui.image(self._cvuiframe, 800 + 15, 40, display_cropping_frame)

        # 文字列、バー：クラス分類結果
        if classifications is not None:
            for i, classification in enumerate(classifications):
                cvui.printf(self._cvuiframe, 800 + 15, 230 + (i * 35), 0.4,
                            0xFFFFFF, classification[1])
                cvui.rect(self._cvuiframe, 800 + 15, 245 + (i * 35),
                          int(181 * float(classification[2])), 12, 0xFFFFFF,
                          0xFFFFFF)

        # カウンター：スコア閾値
        cvui.printf(self._cvuiframe, 800 + 15, 420, 0.4, 0xFFFFFF, 'THRESHOLD')
        cvui.counter(self._cvuiframe, 800 + 95, 414, self._score_threshold,
                     0.1)
        self._score_threshold[0] = max(0, self._score_threshold[0])
        self._score_threshold[0] = min(1, self._score_threshold[0])

        cvui.update()

    def show(self):
        """
        [summary]
          描画
        """
        cvui.imshow(self._window_name, self._cvuiframe)
        if self._window_position is not None:
            cv.moveWindow(self._window_name, self._window_position[0],
                          self._window_position[1])

    def set_score_threshold(self, number):
        """
        [summary]
          検出スコア閾値設定
        """
        self._score_threshold[0] = number

    def get_score_threshold(self):
        """
        [summary]
          検出スコア閾値取得
        """
        return self._score_threshold[0]

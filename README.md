# FingerFrameLens
FingerFrame検出を行った結果に対し、画像クラス分類を行うデモです。<br>
* FingerFrame検出：EfficientDet D0<br>
* 画像クラス分類：EfficientNet B0<br>

# Requirement 
* Tensorflow 2.3.0 or later
* OpenCV 3.4.2 or later

# Demo
デモの実行方法は以下です。
```bash
python main.py
```

また、デモ実行時には、以下のオプションが指定可能です。
* --device<br>カメラデバイス番号の指定 (デフォルト：0)
* --width<br>カメラキャプチャ時の横幅 (デフォルト：960)
* --height<br>カメラキャプチャ時の縦幅 (デフォルト：540)
* --model<br>モデル読み込みパス (デフォルト：'model/EfficientDetD0/saved_model')
* --score_th<br>物体検出閾値 (デフォルト：0.7)
* --fps<br>処理FPS (デフォルト：10.1) ※推論時間がFPSを下回る場合のみ有効

# Reference
[Kazuhito00/FingerFrameDetection-TF2](https://github.com/Kazuhito00/FingerFrameDetection-TF2)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FingerFrameDetection-TF2 is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

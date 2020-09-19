[[Japanese](https://github.com/Kazuhito00/FingerFrameLens)/English] 
# FingerFrameLens
This is a demo that classifies images based on the results of FingerFrame detection.<br>
* FingerFrame Detection：EfficientDet D0<br>
* Image Classification：EfficientNet B0<br>

# Requirement 
* Tensorflow 2.3.0 or later
* OpenCV 3.4.2 or later

# Demo
Here's how to run the demo.
```bash
python main.py
```

The following options can be specified when running the demo.
* --device<br>camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Vertical width at the time of camera capture (Default：540)
* --model<br>Model loading path (Default：'model/EfficientDetD0/saved_model')
* --score_th<br>Detection threshold (Default：0.7)
* --fps<br>Processing FPS (Default：10.1) ※Valid only if the inference time is less than FPS

# Reference
The following trained-model is used to detect FingerFrame.
[Kazuhito00/FingerFrameDetection-TF2](https://github.com/Kazuhito00/FingerFrameDetection-TF2)

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
 
# License 
FingerFrameLens is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# Topic
Using Efficient Net to solve problem:      
Face anti spoofing

# Description
- Based on better result in computer vision tasks, Efficient net will be used in this code. Paper about EfficientNet [here](https://arxiv.org/pdf/1905.11946.pdf)
- Face anti spoofing is a hot topic recently. A very detail survey about this topic [here](https://arxiv.org/pdf/2010.04145.pdf)
- The code may be not clean, I am learning to improve in future.

# Installation     
- Use pip to install all needed library      
```bash
pip install -r requirements.txt
```
# Usage
1. Config and train models
 - Go to model_zoo.py to see my initial models or add some
 - Go to config.py to see initial config, you can change to match with your model
 - Go to train.py to see training process
 - Note that all of the code using tensorflow and keras
2. Evaluate models
 - There are two file can evaluate the model: evaluate_opencv.py and evaluate_tf.py, the biggest difference between them is read image from directory: one read image by opencv, the other read by tensorflow, and then image will be resized to match model input. Both read function and resize function affect the performance of model, so notice that.
 - After run file to evaluate, the result will saved in / result_ \[model_name] / test_ \[cv or tf] . Result will include eer calculation, wrong sample path, and score prediction of test samples.    
 3. Demo
 In demo folder, you can see some files:
  - demo_face_detector.py: read an image and show image with face detection only
  - demo_image.py: read an image, then detect face in image and predict that face is real or spoof 
  - demo_video.py: read a video, show result of each frame in video, you can choose save video result or not
  - demo_camera.py: read from camera of computer, this will show result of each frame
  - gui.py: a simple GUI with tkinter to test on image
 All the files, you can change model_path to change model. The default will use a efficient net b0 which i trained before.
# Note
These are important notes about the code



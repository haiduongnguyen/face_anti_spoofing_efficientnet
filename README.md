# Topic
Using Efficient Net to solve problem:      
Face anti spoofing      
Phase 1: photo attack

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
 - Go to **config.py** , change **work_place** to your workplace, **result_all_model** that all model checkpoint after training will be saved at. Note that Result and Data should be saved outside of workplace for easy maintainance.
 - Data saved like this norm: ![Data folder tree]()
 - Go to **model_zoo.py** to see my initial models or add some of yours
 - Go to **train.py** to see training process, one important thing is the model_name is very importancce, all the code will recognize models by their name.
 - Change model attributes and then run **CUDA_VISIBLE_DEVICES=0 python train.py** if you have gpu
 - Note that all of the code using tensorflow and keras
2. Evaluate models
 - Now **evaluate is in train.py**. After training, checkpoint will be save at ../result/_model_name_/train/checkpoint, then each checkpoint will be evaluate by tensorflow function      
 - Another way to evaluate a checkpoint is run **evaluate_tf** (use tf function) or **evaluate_cv** (use opencv function), you need to change 2 parameters in main():
 	- **model_name**: the code recognize model by name, please remember it exactly      
 	- **index_cp_list**: list of checkpoint want to be evaluated
3. Demo
In demo folder, you can see some files:
- **demo_face_detector.py**: read an image and show image with face detection only
- **demo_image.py**: read an image, then detect face in image and predict that face is real or spoof 
- **demo_video.py**: read a video, show result of each frame in video, you can choose save video result or not
- **demo_camera.py**: read from camera of computer, this will show result of each frame
- **gui.py**: a simple GUI with tkinter to test on image
All the files, you can change model_path to change model. The default will use a efficient net b0 which i trained before.
# Note
These are important notes about the code



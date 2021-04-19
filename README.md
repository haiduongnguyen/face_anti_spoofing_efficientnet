# description
model: efficient net b4     
data: celeb - photo      
- train: 195k images (live + spoof)      
- test == valid: 28k images (live + spoof)      
framework: tensorflow + keras
python: 3.6.9



# Run code
## step 1: create virtual environment    
run        
**python3 -m venv venv**      
**python3 source venv/bin/activate**     
**cd face_anti_spoofing_efficientnet**      

install needed packages      
**pip install -r 'requirement.txt'**  

## step 2: modify parameter in config.py and make needed folder
- work_place : the current directory that we use to work       
- raw_data : path to celeb - photo data     

if you want to change backbone, model compile or image config, change:       
- model config: image_size, image_depth, batch_size, optimizer   
- model_name          
- run **python3 config.py**   
      
## step 3: make helper json 
helper json will modify from original json to right path data   
come to make_helper_json and change the path to work_place in **line 5**. Then run        
**python3 make_json/make_helper_json.py**   
  
## step 4: make data    
from raw data we must cut face from image & save them to another directory
careful that here I use valid set and test set same, if have different valid set and test set, need to init new json and make data again       
run    
**python3 make_data.py**      
  
## step 5: train model  
can modify model in **model_zoo.py** or create new model  
if change model, careful to  size of model input, function call model in train_celeb.py, model_name in config.py  
if you want to use data augmentation, uncomment from line 18 - 24       
if you want to change model complie, change from line 58 - 60
if you want to change compile, callback configuration, change from line 66 - 74
run              
**python3 train_celeb.py**     
to use gpu, add this command before python3 train_celeb.py      
**CUDA_VISIBLE_DEVICES=0**       
after training, result training will be saved in result_training_output.txt
  
## step 6: evaluate model  
evaluate model on test set  
run   
**python3 evaluate.py**     
the result will be saved in result_test.txt

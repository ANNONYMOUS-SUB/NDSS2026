##### To perform efficiency attack on DETR models

```sh
conda create -n teaspoon python=3.9.21
conda activate teaspoon
```

```sh 
export HF_HOME="<your_huggingface_cache_directory>"
export TRANSFORMERS_CACHE="<your_huggingface_cache_directory>"

# for example:

export HF_HOME="~/huggingface_cache"
export TRANSFORMERS_CACHE="~/huggingface_cache"
```

```sh
git clone git@github.com:<replace_this_with_the_actual_name_of_repo>.git && cd <replace_this_with_the_actual_name_of_repo>
git checkout teaspoon
pip install -r requirements.txt

# if:
# ImportError: Can't determine version for bottleneck
conda update pandas
conda remove bottleneck
conda install bottleneck
```

```sh
cd scripts
./baseline.sh
./teaspoon.sh
./teastatic.sh
```

or


```sh
# need to at least specify these two parameters
python ../main.py --model_id <id> --algorithm <a> 

# also other optional parameters
python ../main.py --model_id <id> --algorithm <a> --it_num <n> --val_size <val> --target_idx <t> --output_dir <o_dir> --if_save <s> --save_dir <s_dir>
```


##### To run the pipeline

```sh
conda create -n teapipe python=3.12.9
conda activate teapipe
```

```sh
cd pipeline_traffic
conda install -c conda-forge pygobject
pip install pipeline_requirements.txt
python embed_face.py # prepare a demo dataset for the pipeline
python build_lpr_db.py
```

```sh
alias python=python3 # if necessary
python traffic.py --model_id <id> --algorithm <a>
```

```

                                  ____________ face recognition ________         
                                 /                                      \______ knowledge 
                                /                                       /       retrieval    
data ----- object detection ---|----- license plate segmentation --- ocr             \ 
                                \                                                    |--- language model
                                 \                                                   /
                                  \___ image captioning ____________________________/


object detection:
    - YOLO or Vision Transformer
    - Input: PIL Image
    - Output: (box, cls, scores)
    
face recognition:
    - FaceNet
    - Input: bounding boxes which has a cls label == "person"
    - Output: 
    
license plate recognition:
    - DeepLab V3
    - Input: bounding boxes which has a cls label == "car"
    - Output:
    
ocr:
    - onnx ocr
    - Input: cropped license plate image
    - Output: text
    
image captioning:
    - huggingface "microsoft/git-base"
    - Input: bounding boxes which has a cls label == ["person", "car", "traffic lights", "stop sign"]
    - Output: text
        
knowledge retrieval:
    - Database 1
    - Explanation: query database if exists a license plate, if so, return other info of the vehicle
    - Input: license plate text
    - Output: vehicle info text
    
    - Database 2
    - Explanation: compare the similarity of the detected face with the stored faces
    - Input: face embedding
    - Output: name of the most similar face and confidence score
    
language model:
    - GPT 2/Grok 2 (now offer two choices)
    - Input: 
        maintain a buffer for each stream
        query GPT-2/Grok-2 for content summarization
    - Output: text
    

```
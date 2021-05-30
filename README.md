# POSTECH CSED538 Group12 Project

## Preparation
First of all, I highly recommend you to use "virtualenv" or "anaconda" for managing python libraries in Ubuntu/Mac.

After making your own private python environment using whatever, Please install needed python packages by following : pip install -r requirements.txt

You have to install proper pytorch with your own gpu version; 

## Dataset
### We use DL20, which has 20 classes.
You have to make following dataset hierarchy:

---

dataset / DL20 / [train, valid, test]

---

train - [0, 1, 2, ... , 19]

valid - [0, 1, 2, ..., 19]

test - only have images without labels

---
### semi-sup split 
python split.py

dataset / DL20 / [0.1_label_path_list.txt, ... ]


## Wandb
You have to install wandb; ML recording & visualization tool.

how to start/use wandb : https://greeksharifa.github.io/references/2020/06/10/wandb-usage/

## For training
python3 main.py

## For base line reproduction
download baseline weights : https://drive.google.com/file/d/158uepP38iGlZ_UKCL6PJQ39RTfbYhUcH/view?usp=sharing

python3 main.py --pretrained-ckpt="./YOUR_PATH/baseline.pth"

## For pseudo simclr 

### First, prepare psuedo labeling file 
1. train model supervised manner using some labeled data(ratio: 0.125, 0.5 ...)
2. using trained model, generate pseudo label about unlabeled data (Refer 0.125_unlabel_path_list2.txt)
    ex) each line consist of unlabeled-datapath & psuedo-label
3. training simclr using pseudo label: python3 z_semi_kam_run.py 

## Basic informations about framework
### backbone : efficientnet-b04 

## Contributorse
감제원 20202637    
권동현 20212423     
김민석 20182698    
이주용 20160271

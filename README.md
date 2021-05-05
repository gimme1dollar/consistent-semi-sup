# CSED538 Group12 Project

## Preparation
First of all, I highly recommend you to use "virtualenv" or "anaconda" for managing python libraries.

After making your own private python environment using whatever, Please install needed python packages by following : pip install -r requirements.txt

## Dataset
We use DL20, which have 20 classes.
You have to make following dataset hierarchy:

dataset / DL20 / [train, valid, test]

---

train - [0, 1, 2, ... , 20]

valid - [0, 1, 2, ..., 20]

test - only have images without labels



## Wandb
You have to install wandb; ML record & visualization tool.

how to start/use wandb : https://greeksharifa.github.io/references/2020/06/10/wandb-usage/

## For training
python3 main.py

## Basic informations about framework
### backbone : efficientnet-b04 

## Contributors
감제원 20202637    
권동현 20212423     
김민석 20182698    
이주용 20160271

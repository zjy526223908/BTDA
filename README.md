# BTDA
implement for paper Rethinking Domain Adaptation Blending target Domain Adaptation by Adversarial Meta Adaptation Network 

## Requirements
* python 2.7
* PyTorch 0.4.1
* cuda 8.0
* numpy
* tqdm
* visdom
* PIL
* argparse
* sklearn
* torchvision

## download dataset
cd BTDA
bash download_dataset.sh

## digit dataset
cd BTDA/Digit
python train.py --source_name mnist --gpu_id 0

## office31 dataset
cd BTDA/Office
python alexnet_train.py --dataset_name Office31 --source_name amazon --gpu_id 0
python resnet_train.py  --dataset_name Office31 --source_name amazon --gpu_id 0

## officeHome dataset
cd BTDA/Office
python alexnet_train.py --dataset_name OfficeHome --source_name A --gpu_id 0
python resnet_train.py  --dataset_name OfficeHome --source_name A --gpu_id 0

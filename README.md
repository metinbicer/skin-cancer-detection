# Overview
Udacity Deep Learning Nanodegree course project on skin cancer detection paper (https://www.nature.com/articles/nature21056)

## About
It uses transfer learning to classify three classes of skin cancer. Different pretrained models in Pytorch can be selected. Data loading will be based on selected model. Pretrained model can be trained and tested. New predictions can be made by the new model.

## Usage
1. Clone the repo
```
git clone https://github.com/metinbicer/skin-cancer-detection
cd skin-cancer-detection
```
 2. Download the datasets
```
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip
unzip *.zip
```
 3. Store them in respective folders
```
mkdir data 
mv train.zip valid.zip test.zip data
```
4. Dependencies

Install the following packages
* pytorch
* torchvision
* numpy
* matplotlib
* PIL
5. Architecture and parameters

You can change the architecture and hyperparameters within the code

6. Run
```
python main.py
```
7. Note

I could obtain 66% accuracy in testing with very little computational resources. I will try to update the best model.

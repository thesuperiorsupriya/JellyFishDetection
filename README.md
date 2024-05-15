# JellyFishDetection
The aim of this project is to develop a computer vision system capable of detecting type of jellyfish in underwater images and videos. This system will assist in monitoring jellyfish populations and their movements, which is crucial for ecological studies, marine safety, and tourism management.
Data Collection:

Gather a diverse dataset of underwater images and videos featuring various species of jellyfish in different environments and lighting conditions.
Annotate the dataset with bounding boxes or segmentation masks to label the jellyfish instances accurately.
Preprocessing:

Enhance image quality using techniques like contrast adjustment, noise reduction, and color correction to handle underwater image challenges.
Normalize and resize images to a consistent format suitable for model training.
Model Selection:

YOLO (You Only Look Once) is used in the model.

Training and Validation:

Split the dataset into training, validation, and test sets ensuring a balanced distribution of jellyfish species and environmental conditions.
Train the selected models using the training set while tuning hyperparameters to improve performance.
Validate the models on the validation set to monitor for overfitting and adjust training procedures accordingly.
The Classes of JellyFish used are 
1. Moon JellyFish
2. Mauve Stinger Jelly fish
3. Lions Mane Jelly Fish
4. Compass Jelly Fish
5. Blue Jelly Fish
6. Barrel Jelly fish

# PROPOSED SYSTEM

1] Study basics of machine learning and image recognition.

2] Start with implementation

    A. Front-end development
    B. Back-end development
3] Testing, analyzing and improvising the model. An application using python IDLE and its machine learning libraries will be using machine learning to identify whether a given Banana is rotten or not.

4] use datasets to interpret the object and suggest whether a given Banana on the camera’s viewfinder is rotten or not.

Jetson Nano Compatibility
• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.


Installation
Initial Configuration
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
Create Swap
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
Cuda env in bashrc
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
Update & Upgrade
sudo apt-get update
sudo apt-get upgrade
Install some required Packages
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
Install Torch
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
Install Torchvision
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
Clone Yolov5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
Download weights and Test Yolov5 Installation on USB webcam
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
Banana Dataset Training
We used Google Colab And Roboflow
train your model on colab and download the weights and pass them into yolov5 folder.
Running Helmet Detection Model
source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
### Demo


 

https://github.com/thesuperiorsupriya/JellyFishDetection/assets/154125788/576877d5-8e13-4555-b31e-aa33c3736df2



https://github.com/thesuperiorsupriya/JellyFishDetection/assets/154125788/576877d5-8e13-4555-b31e-aa33c3736df2





FUTURE SCOPE
• As we know technology is marching towards automation so this project is one of the step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.


• The model is efficient and highly accurate and hence reduces the workforce required.

Reference
1] Roboflow:- https://roboflow.com/
2] Google images
   

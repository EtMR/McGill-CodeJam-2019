# McGill-CodeJam-2019

--- Introduction ---
This repository includes the source code for our work on a Real-time Gesture Detection system (RtGD). The work is developed with the help of my teammates (Kaiwen Xu, Tom Sarry, and Elsa Emilien) during the McGill CodeJam 2019 event (https://devpost.com/software/mr-bean) which wins the second place award in the Optional Theme - Internet of Things. I would also want to acknowledge Brenner Heintz (https://github.com/athena15/project_kojak) for his pre-trained hand-gesture model and the inspiration on this project.

This repository will be structured as follow:
    1. Our goal for the project.
    2. Functions for each script.
    3. Results of the RtGD model.
    4. Future work and reflections.

Note, the code provided here is just the computer vision part of the CodeJam project. For the detailed project description and source code, please refer to the devpost page provided above.

--- 1. Our goal for the project ---
The goal for this project is to develop a RtGD system **having minimum effect on environmental noise**. When compared to the work done by Brenner Heintz, our system is targeted to function properly outside of a dark room. This is tackled by the compibation of a motion detection algorithm (OpenCV) + model retraining on noisy data in our approach. We followed the pretrained model from Brenner's repo and targeted the recognition of the same five gestures, fist, L, okay, palm, and peace, in this project. Examples for the retraining gesture data are shown in the figure below: 

![(1) fist](/images/fist.jpg)
![(2) L](/images/L.jpg)
![(3) okay](/images/okay.jpg)
![(4) palm](/images/palm.jpg)
![(5) peace](/images/peace.jpg)

--- 2. Functions for each script ---
In this section, I will introduce the function for each scripts and zip file based on my experimental flow. A illustration for the experimental flow is presented below:

>>>  pretrain VGG model (/models.zip/VGG_cross_validated.h5)
-->  record images for downstream task (1_Record_Image.py) (NewTraining20191216.zip)
-->  retrain the model (2_Retrain model.py) (retrained_20200506.h5)
-->  operate the RtGD system (3_RtGD.py)

The pretrained and retrained models are stored in the model.zip file. If you are interested to record new images for your downstream task, the script provided in (1_Record_Image.py) can serve well for this perpose. The recording script includes a motion detection algorithm which only catch up moving parts and omit environmental noise (ex: lights from the window or lamp). An example for the recorded images, with 20 images for each classes, are stored in NewTraining20191216.zip. These images are then applied to retrain our model using script (2_Retrain model.py). Due to the time limitation in CodeJam competition, we frozen the convolution layer of the VGG model and use the recorded images to only retrain the dense layer which have 3252997 trainable parameters.

After retraining on our own task, real-time inference can be operated by running the script (3_RtGD.py); the same motion detection algorithm is also used here. Due to the retraining process, our model is able to recognize different hand gestures with minimum interference from the environmental light noise. A testing accuracy of 85% is achieved by using the recorded dataset.

--- 3. Results of the RtGD model ---
The result of running the RtGD model is illustrated in the following figure

--- 4. Future work and reflections ---

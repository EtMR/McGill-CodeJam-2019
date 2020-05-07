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


--- 3. Results of the RtGD model ---
--- 4. Future work and reflections ---

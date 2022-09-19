# Rock-Scissor-Paper-move-recognition
In this project, I use classification ML to detect Rock Scissor or Paper through the camera. The figure represents the workflow: 
[insert fig. ]


With Touchdesigner, you can play Rock Scissor Paper against the CPU (random move). 


# 1) EXTRACT HAND LANDMARKS WITH MEDIAPIPE
I use Mediapipe to extract hand landmarks. Mediapipe detects hands on the screen in real time and transforms them in [x, y, z] coordinates. 
In order to train the algorithm, i used three categories of pictures, representing respectively people doing Rock, Scissor or Paper. 

*hand_detection.py* contains the script that transforms pictures 
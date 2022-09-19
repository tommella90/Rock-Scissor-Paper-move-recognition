# Rock-Scissor-Paper-move-recognition
In this project, I use classification ML to detect Rock Scissor or Paper through the camera. The figure represents the workflow: 
[insert fig. ](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/tree/main/img)
1) I collected images of people distinguishing Rock, Scissor and Paper. 
2) I transformed the images in landmarsk (coordinates) with Mediapipe. 
3) I applied ML algorightms on the new data to train a model able to recognize the hand-move
4) I apply the model to new data (video-camera) and detect in real time the current hand-move. 


With Touchdesigner, you can play Rock Scissor Paper against the CPU (random move). 


# 1) EXTRACT HAND LANDMARKS WITH MEDIAPIPE
I use Mediapipe to extract hand landmarks. Mediapipe detects hands on the screen in real time and transforms them in [x, y, z] coordinates. 
In order to train the algorithm, i used three categories of pictures, representing respectively people doing Rock, Scissor or Paper. 

*hand_detection.py* contains the script that transforms pictures 
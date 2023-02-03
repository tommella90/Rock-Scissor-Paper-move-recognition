 # ROCK SCISSOR PAPER MOVE RECOGNITION 
 ### thanks to TOUCHDESIGNER and MEDIAPIPE
https://user-images.githubusercontent.com/66441052/216642513-c82ba289-7721-4d4c-ada8-5e7b69df2de1.mp4

If you want to try it: 
- clone the repo 
```
git clone https://github.com/tommella90/Rock-Scissor-Paper-move-recognition.git
```
- create a conda env called td_py (necessary for touchdesigner): 
```
conda create -n td_py python=3.9
```
- and run the requirements:
```
pip install requirements.txt
```

## DESCRIPTION
In this project, I use classification ML to detect Rock Scissor or Paper through the camera. The figure represents the workflow: 

1) I collected images of people distinguishing Rock, Scissor and Paper. 
2) I transformed the images in landmarsk (coordinates) with Mediapipe. 
3) I applied ML algorightms on the new data to train a model able to recognize the hand-move
4) I apply the model to new data (video-camera) and detect in real time the current hand-move. 
Finally, With Touchdesigner, you can play Rock Scissor Paper against the CPU (random move). 

<img align="center" width="700" height="400" src="https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/img/workflow.png">


## EXTRACT HAND LANDMARKS WITH MEDIAPIPE
I use Mediapipe to extract hand landmarks. Mediapipe detects hands on the screen in real time and transforms them in [x, y, z] coordinates. 
In order to train the algorithm, i used three categories of pictures, representing respectively people doing Rock, Scissor or Paper. 

[This file](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/hand_detection.py) (*hand_detection.py*) contains the script that transforms pictures into landmarsk. For obvious reasons, I deleted the pics from family and friends and ketp only pics of me (I don't mind) and the ones from the Kaggle database (insert link). The excel file with the data (already transformed in landmarks), [is here](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/data/hands_coords.csv)

https://user-images.githubusercontent.com/66441052/191137387-ca81db3e-a1fe-4846-92fa-c686a45b440f.mp4

## TRAIN THE MODEL 
I used the landmarks to train a model able to recognize the hand-move on new data. The script is [here](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/hand_modeling.py).
Note that, despite the accuracy is always very high (~99%) it might not be accurate. This happens because the train dataset might be non-representative. I used a total of ~5000 pictures and the model worked pretty well. Among all the models used, Random Forest performs the best. 

## APPLY THE MODEL IN TOUCHDESIGNER
I saved the models in pickle and uploaded them in Touchdesigner. The extension script is [here](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/RSC_class.py). It contains several features: 
- Model: allows to choose among the models trained
- Select input: allows you to see the camera or the Mediapipe Landmarks transformed in a hand 3D model
- Input data: see the landmarsk from the database or the real-time camera
- Game mode: switch the game mode ON/OFF
- Name: select your name before playing tha game
- Reset Game: reset game score 

If game mode is OFF, you can simply input a move on the camera and see the result. 
(see video at the top)

## PLAY ROCK SCISSOR PAPER AGAINST THE CPU
If you switch game mode ON, you will see the score and you will be able to input your move. The CPU move is 100% random. Note that, since the move detection cannot be 100% accurate, to input your move you need to hold it for a couple of seconds (watch the colored bars on the top-left). Here is the script that manages the Game on Touchdesigner: [link](https://github.com/tommella90/Rock-Scissor-Paper-move-recognition/blob/main/game_script.py)

https://user-images.githubusercontent.com/66441052/216642651-c1c33c2d-7a53-4d5e-81bd-807829d02a22.mp4



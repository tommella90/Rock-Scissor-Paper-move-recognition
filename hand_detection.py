#########################
### HAND DETECTION CV ###
#########################
## import libraries
import cv2
import math
import numpy as np
import pandas as pd
import os
import pickle

## import mediapipe (hands detection)
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

## FIND DISTANCES
## Adding distances among some fingers as additional features
def find_landmark_distances(hand_dictionary):
    """Function to find the landmarks and landmark distnaces """
    ## wrist (palm)
    wrist_x, wrist_y, wrist_z = hand_dictionary['WRIST_x'], hand_dictionary['WRIST_y'], hand_dictionary['WRIST_z']
    middle_x, middle_y, middle_z =  hand_dictionary['MIDDLE_FINGER_TIP_x'], hand_dictionary['MIDDLE_FINGER_TIP_y'], hand_dictionary['MIDDLE_FINGER_TIP_z']
    thumb_x, thumb_y, thumb_z =  hand_dictionary['THUMB_TIP_x'], hand_dictionary['THUMB_TIP_y'], hand_dictionary['THUMB_TIP_z']
    pinky_x, pinky_y, pinky_z =  hand_dictionary['PINKY_TIP_x'], hand_dictionary['PINKY_TIP_y'], hand_dictionary['PINKY_TIP_z']
    ring_x, ring_y, ring_z =  hand_dictionary['RING_FINGER_TIP_x'], hand_dictionary['RING_FINGER_TIP_y'], hand_dictionary['RING_FINGER_TIP_z']
    middle_x2, middle_y2, middle_z2 =  hand_dictionary['MIDDLE_FINGER_MCP_x'], hand_dictionary['MIDDLE_FINGER_MCP_y'], hand_dictionary['MIDDLE_FINGER_MCP_z']
    pinky_x2, pinky_y2, pinky_z2 =  hand_dictionary['PINKY_MCP_x'], hand_dictionary['PINKY_MCP_y'], hand_dictionary['PINKY_MCP_z']

    ## index - middle vs wrist
    x1 = (hand_dictionary['INDEX_FINGER_TIP_x'] + hand_dictionary['MIDDLE_FINGER_TIP_x']) / 2
    y1 = (hand_dictionary['INDEX_FINGER_TIP_y'] + hand_dictionary['MIDDLE_FINGER_TIP_y']) / 2
    z1 = (hand_dictionary['INDEX_FINGER_TIP_z'] + hand_dictionary['MIDDLE_FINGER_TIP_z']) / 2
    dist_sq1 = (((x1 - wrist_x)**2) + ((y1 - wrist_y)**2) + ((z1 - wrist_z)**2)**(1/2))

    ## ring - pinky vs wrist
    x2 = (hand_dictionary['RING_FINGER_TIP_x'] + hand_dictionary['PINKY_TIP_x']) / 2
    y2 = (hand_dictionary['RING_FINGER_TIP_y'] + hand_dictionary['PINKY_TIP_y']) / 2
    z2 = (hand_dictionary['RING_FINGER_TIP_z'] + hand_dictionary['PINKY_TIP_z']) / 2
    dist_sq2 = (((x2 - wrist_x)**2) + ((y2 - wrist_y)**2) + ((z2 - wrist_z)**2)**(1/2))

    ## thumb vs wrist
    thumb_x = hand_dictionary['THUMB_TIP_x']
    thumb_y = hand_dictionary['THUMB_TIP_y']
    thumb_z = hand_dictionary['THUMB_TIP_z']
    dist_sq3 = (((thumb_x - wrist_x)**2) + ((thumb_y - wrist_y)**2) + ((thumb_z - wrist_z)**2)**(1/2))

    ## middle vs pinky
    dist_sq4 = (((middle_x - pinky_x)**2) + ((middle_y - pinky_y)**2) + ((middle_z - pinky_z)**2)**(1/2))

    ## thumb vs pinky
    dist_sq5 = (((middle_x - pinky_x)**2) + ((middle_y - pinky_y)**2) + ((middle_z - pinky_z)**2)**(1/2))

    ## thumb vs middle
    dist_sq6 = (((middle_x - thumb_x)**2) + ((middle_y - thumb_y)**2) + ((middle_z - thumb_z)**2)**(1/2))

    ## ring vs middle
    dist_sq7 = (((middle_x - ring_x)**2) + ((middle_y - ring_y)**2) + ((middle_z - ring_z)**2)**(1/2))

    ## pinky bottom - top 
    dist_sq8 = (((pinky_x - pinky_x2)**2) + ((pinky_y - pinky_y2)**2) + ((pinky_x - pinky_z2)**2)**(1/2))

    ## middle bottom - top 
    dist_sq9 = (((middle_x - middle_x2)**2) + ((middle_y - middle_y2)**2) + ((pinky_x - middle_z2)**2)**(1/2))

    hand_dictionary['indmid_wrist'], hand_dictionary['ringpinky_wrist'], hand_dictionary['thumb_wrist'] = dist_sq1, dist_sq2, dist_sq3
    hand_dictionary['middle_pinky'], hand_dictionary['thumb_pinky'], hand_dictionary['thumb_middle'] = dist_sq4, dist_sq5, dist_sq6
    hand_dictionary['middle_ring'], hand_dictionary['pinky_itslef'], hand_dictionary['middle_itself'] = dist_sq7, dist_sq8, dist_sq9

    return hand_dictionary


## standardization
def Standardization(Series):
    mean_series = Series.mean()
    std_series = Series.std()
    Series = (Series - mean_series) / std_series
    return Series


## def load file
def load(filename = "filename.pickle"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("File not found!")


## def resize image (standardize values)
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize(image):
    h, w = image.shape[:2]
    if h < w:
        cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))


## function to extract the coordinates and populate a df
def GetHandCoords(directory):
    i = 0
    df = pd.DataFrame()

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        for file in os.listdir(directory):
            i += 1
            print(i)
            img = cv2.imread(directory + file , 1)
            resize(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.flip(img, 1)

            results = hands.process(img)

            image_hight, image_width, _ = img.shape
            annotated_image = img.copy()  ## to use to draw landmarks on the image

            try:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_dictionary = {
                        'WRIST_x' : hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                        'WRIST_y' : hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                        'WRIST_z' : hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,

                        'THUMB_CMC_x' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
                        'THUMB_CMC_y' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
                        'THUMB_CMC_z' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,

                        'THUMB_MCP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                        'THUMB_MCP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                        'THUMB_MCP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z,

                        'THUMB_IP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                        'THUMB_IP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
                        'THUMB_IP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z,

                        'THUMB_TIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        'THUMB_TIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                        'THUMB_TIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,

                        'INDEX_FINGER_MCP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                        'INDEX_FINGER_MCP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                        'INDEX_FINGER_MCP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z,

                        'INDEX_FINGER_PIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                        'INDEX_FINGER_PIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                        'INDEX_FINGER_PIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z,

                        'INDEX_FINGER_DIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                        'INDEX_FINGER_DIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                        'INDEX_FINGER_DIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z,

                        'INDEX_FINGER_TIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        'INDEX_FINGER_TIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        'INDEX_FINGER_TIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z,

                        'MIDDLE_FINGER_MCP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                        'MIDDLE_FINGER_MCP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                        'MIDDLE_FINGER_MCP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z,

                        'MIDDLE_FINGER_PIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                        'MIDDLE_FINGER_PIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                        'MIDDLE_FINGER_PIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z,

                        'MIDDLE_FINGER_DIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                        'MIDDLE_FINGER_DIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                        'MIDDLE_FINGER_DIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z,

                        'MIDDLE_FINGER_TIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                        'MIDDLE_FINGER_TIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                        'MIDDLE_FINGER_TIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z,

                        'RING_FINGER_MCP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                        'RING_FINGER_MCP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                        'RING_FINGER_MCP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,

                        'RING_FINGER_PIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                        'RING_FINGER_PIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                        'RING_FINGER_PIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z,

                        'RING_FINGER_DIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                        'RING_FINGER_DIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                        'RING_FINGER_DIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z,

                        'RING_FINGER_TIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                        'RING_FINGER_TIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                        'RING_FINGER_TIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z,

                        'PINKY_MCP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                        'PINKY_MCP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                        'PINKY_MCP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z,

                        'PINKY_PIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                        'PINKY_PIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                        'PINKY_PIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z,

                        'PINKY_DIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
                        'PINKY_DIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
                        'PINKY_DIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z,

                        'PINKY_TIP_x' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                        'PINKY_TIP_y' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                        'PINKY_TIP_z' : hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z,

                        'hand': results.multi_handedness[0].classification[0].label,
                        'score': results.multi_handedness[0].classification[0].score,
                        'move': os.path.basename(os.path.normpath(directory))

                    }

                find_landmark_distances(hand_dictionary)
                row = pd.DataFrame.from_dict(hand_dictionary, orient='index').T
                df = pd.concat([df, row])

            except Exception as inst:
                print(inst.args)

        return df

## function
def ApplyModel(df, model):
    move = model.predict(df)
    return

#img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## GET CAMERA INPUT AND EXTRACT THE MOVE
model = load("../model.pickle")
print(model)


#%%
# RUN MEDIAPIPE HANDS AND CREATE THE DATAFRAMES
#mp_hands.Hands()

## path for images and dataframe
from os import listdir
s_path = 'data/scissor/'
r_path = 'data/rock/'
p_path = 'data/paper/'

s = os.listdir(s_path)
r = os.listdir(r_path)
p = os.listdir(p_path)

scissor = pd.DataFrame()
for i in s:
    print(f'{s_path}' + f'{i}/')
    data = GetHandCoords(s_path + f"{i}/")
    scissor = pd.concat([scissor, data])

rock = pd.DataFrame()
for i in r:
    print(f'{r_path}' + f'{i}/')
    data = GetHandCoords(r_path + f"{i}/")
    rock = pd.concat([rock, data])

paper = pd.DataFrame()
for i in p:
    print(f'{p_path}' + f'{i}/')
    data = GetHandCoords(p_path + f"{i}/")
    paper = pd.concat([paper, data])

## concatenate and save the dataframe
df = pd.concat([scissor, rock, paper])
df.to_csv('data/hand_coords_incomplete.csv', index = False, encoding='utf-8')
print('done')


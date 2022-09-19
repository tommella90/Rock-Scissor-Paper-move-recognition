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

## FIND RELATIVE POSITIONS (distances)
## relative position
def RelativePositions(hand_dictionary):
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
            #print(results.multi_handedness)
            #results.multi_handedness

            image_hight, image_width, _ = img.shape
            annotated_image = img.copy()

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

                RelativePositions(hand_dictionary)
                row = pd.DataFrame.from_dict(hand_dictionary, orient='index').T
                df = pd.concat([df, row])

            except Exception as inst:
                print(inst.args)

        return df


def CenterCoordinates(df):
    x_list, y_list, z_list = [], [], []

    for i in df.columns:
        if "_x" in i:
            x_list.append(i)
        elif "_y" in i:
            y_list.append(i)
        elif "_z" in i:
            z_list.append(i)

    x_vals = df[x_list]
    x_vals = x_vals - x_vals.iloc[0,0]
    y_vals = df[y_list]
    y_vals = y_vals - y_vals.iloc[0,0]
    z_vals = df[z_list]
    z_vals = z_vals - z_vals.iloc[0,0]
    others = df[['indmid_wrist', 'ringpinky_wrist', 'thumb_wrist', 'middle_pinky',
                 'thumb_pinky', 'thumb_middle', 'middle_ring', 'pinky_itslef', 'middle_itself']]

    df_new = pd.merge(x_vals, y_vals, left_index=True, right_index=True)
    df_new = pd.merge(df_new, z_vals, left_index=True, right_index=True)
    df_new = pd.merge(df_new, others, left_index=True, right_index=True)

    return df_new



## function
def ApplyModel(df, model):
    move = model.predict(df)
    return


## OPEN IMAGE
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize(image):
    h, w = image.shape[:2]
    if h < w:
        cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

#img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## GETO CAMERA INPUT AND EXTRACT THE MOVE
model = load("../data/model.pickle")
print(model)


#%%
# RUN MEDIAPIPE HANDS AND CREATE THE DATAFRAMES
mp_hands.Hands()

from os import listdir
s_path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/scissor/'
r_path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/rock/'
p_path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/paper/'

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

scissor_center = CenterCoordinates(scissor)
rock_center = CenterCoordinates(rock)
paper_center = CenterCoordinates(paper)

'''
scissors1 = GetHandCoords(path + 'scissor_kaggle/')
paper1 = GetHandCoords(path + 'rock_kaggle/')
rock1 = GetHandCoords(path + 'paper_kaggle/')
scissors2 = GetHandCoords(path + 'scissor_me/')
paper2 = GetHandCoords(path + 'rock_me/')
rock2 = GetHandCoords(path + 'paper_me/')
scissors3 = GetHandCoords(path + 'scissor_others/')
paper3 = GetHandCoords(path + 'rock_others/')
rock3 = GetHandCoords(path + 'paper_others/')
'''

#df_2 = pd.concat([scissor, paper, rock])
df_center = pd.concat([scissor_center, rock_center, paper_center])
#df2 = pd.concat([df1, scissors2, paper2, rock2])
#df3 = pd.concat([df2, scissors2, paper2, rock2])

path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/'
#df_2.to_csv(path + 'hand_coords_level2.csv', index = False, encoding='utf-8') # False: not include index
df_center.to_csv(path + 'hand_coords_center.csv', index = False, encoding='utf-8') # False: not include index

#df2.to_csv(path + 'hand_coords2.csv', index = False, encoding='utf-8') # False: not include index
#df3.to_csv(path + 'hand_coords_all.csv', index = False, encoding='utf-8') # False: not include index


#%% TRY OPEN FILES IN LOOP
'''
directory = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/try/'
for file in os.listdir(directory):
    print(file)
    img = cv2.imread(directory + file , 1)
    resize(img)
    cv2.imshow('hand', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
'''
model = load("../data/model.pickle")

from os import listdir
#path = 'C:/Users/tomma/Videos/VideoProc Converter/WIN_20220725_15_21_01_Pro/'
path = "C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/rock/rock_me/"
my_files = listdir(path)
my_files.sort()
pics = []
for i in my_files:
    pics.append(i)
pics.remove(pics[len(pics)-1])
pics

for pic in pics:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        img = cv2.imread(path + pic, 1)
        resize(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        results = hands.process(img)
        #print(results.multi_handedness)
        #results.multi_handedness

        image_hight, image_width, _ = img.shape
        annotated_image = img.copy()

        if not results.multi_hand_world_landmarks:
            continue

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

                }
            coords = pd.DataFrame.from_dict(hand_dictionary, orient='index').T
            model_predict = model.predict(coords)

            if model_predict == 0:
                print('SCISSORS')
            elif model_predict == 1:
                print('PAPER')
            elif model_predict == 2:
                print('ROCK')

        except:
            print("Pic not found")



#%%
path = 'C:/Users/tomma/Pictures/Camera Roll/'
pic = 'WIN_20220719_16_25_04_Pro.jpg'

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    img = cv2.imread(path + pic, 1)
    resize(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)

    results = hands.process(img)
    #print(results.multi_handedness)
    #results.multi_handedness

    image_hight, image_width, _ = img.shape
    annotated_image = img.copy()

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

            }

RelativePositions(hand_dictionary)
hand_dictionary

#%%






#%%
# RUN MEDIAPIPE HANDS AND CREATE THE DATAFRAMES
mp_hands.Hands()

oschdir = 'C:/Users/tomma/Videos/VideoProc Converter/WIN_20220725_15_21_01_Pro/'
from os import listdir
folder = 'C:/Users/tomma/Videos/VideoProc Converter/WIN_20220725_15_21_01_Pro/'
GetHandCoords(folder)






#%%
GetHandCoords("C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/rock/rock_angelo/")


#%% FOR A SINGLE PICTURE

file = "C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/rock/rock_nelson/005.png"

def GetHandCoords(file):
    i = 0
    df = pd.DataFrame()

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        img = cv2.imread(file, 1)
        resize(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)

        results = hands.process(img)
        #print(results.multi_handedness)
        #results.multi_handedness

        image_hight, image_width, _ = img.shape
        annotated_image = img.copy()

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

                }

            RelativePositions(hand_dictionary)
            row = pd.DataFrame.from_dict(hand_dictionary, orient='index').T
            df = pd.concat([df, row])

        except Exception as inst:
            print(inst.args)

        return df

pic = GetHandCoords(file)


#%%
x_list, y_list, z_list = [], [], []
for i in pic.columns:
    if "_x" in i:
        x_list.append(i)
    elif "_y" in i:
        y_list.append(i)
    elif "_z" in i:
        z_list.append(i)

x_vals = pic[x_list]
x_vals = x_vals - x_vals.iloc[0,0]
y_vals = pic[y_list]
y_vals = y_vals - y_vals.iloc[0,0]
z_vals = pic[z_list]
z_vals = z_vals - z_vals.iloc[0,0]
others = pic[['indmid_wrist', 'ringpinky_wrist', 'thumb_wrist', 'middle_pinky',
              'thumb_pinky', 'thumb_middle', 'middle_ring', 'pinky_itslef', 'middle_itself']]
print(list(others))

df = pd.merge(x_vals, y_vals, left_index=True, right_index=True)
df = pd.merge(df, z_vals, left_index=True, right_index=True)
df = pd.merge(df, others, left_index=True, right_index=True)

print(len(df.columns))
#%%

df = CenterCoordinates(pic)

len(df.columns)



#%%
from os import listdir
try_path = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/try/'

t = os.listdir(try_path)
try_all  = pd.DataFrame()
for i in t:
    print(f'{try_path}' + f'{i}/')
    data = GetHandCoords(try_path + f"{i}/")
    try_all = pd.concat([data, try_all])

#scissor = CenterCoordinates(scissor)

#%%


#%% WEBCAM DETECTION
import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


#%% For static images: (1 hand)
import cv2
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print('ld')

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    #file = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/sample_pics/Image from iOS (1).jpg'
    file = 'C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/scissor/scissor_kaggle/1VRzspyXpQ6A2rKy.png'
    image = cv2.imread(file)
    #image = cv2.flip(cv2.imread(image), 1)

# Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if results.multi_hand_landmarks:
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()


    for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('img', annotated_image)
    cv2.waitKey(50000)
    cv2.destroyAllWindows()

scissors1 = GetHandCoords("C:/Users/tomma/Documents/data_science/berlin/final_project/data/hands/try/t2/")
np.array(scissors1.iloc[1,:])
#%%

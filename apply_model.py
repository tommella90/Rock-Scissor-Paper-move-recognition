

# me - this DAT
# scriptOp - the OP which is cooking
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")



mp_hands = mp.solutions.hands

show_move = op('hand_move')


## FIND RELATIVE POSITIONS  (distances between fingers)
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

## lower columns
def LowerColumns(df):
    for i in df.columns:
        df = df.rename(columns = {f"{i}": f"{i.lower()}"})
    return df

def DropFeatures(df):
    df = df.drop(columns=['hand'])
    return df

## replace cat with number
def CleanDataframe(df):
    df['hand'] = df['hand'].replace({"Right": 1, "Left": 0})
    return df

## load file - to load the model trained 
def load(filename = "filename.pickle"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("File not found!")


## UPLOAD THE MODEL
import os
from os import listdir

if op('model_selected')['model'] == 0: 
	model = load("models/tree.pickle")
elif op('model_selected')['model'] == 1:
	model = load("models/knn.pickle")
elif op('model_selected')['model'] == 2:
	model = load("models/rfc.pickle") 
elif op('model_selected')['model'] == 3:
	model = load("models/sgd.pickle") 
elif op('model_selected')['model'] == 4:
	model = load("models/svc.pickle") 


## resize image
def resize(image):
    h, w = image.shape[:2]
    if h < w:
        cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))


## clean df
def CleanDataframe(df):
    df['move'] = df['move'].replace({"scissors": 0, "paper": 1, "rock": 2})
    df['move'] = df['move'].replace({"scissors2": 0, "paper2": 1, "rock2": 2})
    df['hand'] = df['hand'].replace({"Right": 1, "Left": 0})
    df = df[df['score']>0.95]

    return df


# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return

## open mediapipe
hand_mesh = mp_hands.Hands(
	static_image_mode = True,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
    )


def onCook(scriptOp):
	"""Define what happens every frame"""

	# transform the image into data
	img = scriptOp.inputs[0].numpyArray(delayed=True)
	frame = img*255
	frame = frame.astype(np.uint8) #it's the casting of float32 to uint8
	frame = cv2.cvtColor(frame,cv2.COLOR_RGBA2RGB)
	results = hand_mesh.process(frame) 

	# USE MEDIAPPIPE TO GET THE HAND LANDMARKS (per each frame)
	try:
		for hand_landmarks in results.multi_hand_landmarks:

			hand_dictionary = {
				'WRIST_x' : hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
				'WRIST_y' : hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
				'WRIST_z' : hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z,

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
			}
			op('avg_position').par.value0 = hand_dictionary["MIDDLE_FINGER_TIP_x"] +.1
			op('avg_position').par.value1 = hand_dictionary["MIDDLE_FINGER_TIP_y"] 


			# extract distances between fingers
			hand_dictionary = RelativePositions(hand_dictionary)
			
			## clean the dataframe
			df = pd.DataFrame.from_dict(hand_dictionary, orient='index').T
			df = LowerColumns(df)
			df = DropFeatures(df)

			# store the dataframe in touchdesigner
			op('real_time').unstore('*')
			op('real_time').store('df', df.iloc[0,:] )

			# drop the landmarks and keep the distances
			df = df[['indmid_wrist', 'ringpinky_wrist', 'thumb_wrist','middle_pinky', 'thumb_pinky', 'thumb_middle', 'middle_ring','pinky_itslef', 'middle_itself']]

			# APPLY THE MODEL TO THE CURRENT FRAME (predict the gesture)
			model_predict = model.predict((np.array(df.iloc[0,:])).reshape(1,-1))
			probs = model.predict_proba((np.array(df.iloc[0,:])).reshape(1,-1))
			
			# show class probabilitie
			op('probs')[1,1] = round(probs[0][2],2)
			op('color_rock').par.value0 = round(probs[0][2],2)

			op('probs')[1,2] = round(probs[0][0],2)
			op('color_scissor').par.value0 = round(probs[0][0],2)

			op('probs')[1,3] = round(probs[0][1],2) 
			op('color_paper').par.value0 = round(probs[0][1],2)

			list_probs = [
			["rock", round(probs[0][2],2), 0], 
			["scissor", round(probs[0][0],2), 1],
			["paper", round(probs[0][1],2), 2]
			]

			df_probs = pd.DataFrame(list_probs, columns=['moves', 'probs', 'label'])
			df_probs = df_probs.sort_values(by="probs", ascending=False)
			op('most_likely_move').par.value0 = df_probs.iloc[0,2]

			## START ROCK SCISSOR PAPER GAME
			MOVE = op('gesture')
			if op('wait')['v1'] < 0.1:

				if model_predict == 0:
					show_move.par.text = "SCISSORS"	
					MOVE.par.value0 = MOVE.par.value0 + 0.1

				elif model_predict == 1:
					show_move.par.text = "PAPER"	
					MOVE.par.value1 = MOVE.par.value1 + 0.1

				elif model_predict == 2:
					show_move.par.text = "ROCK"
					MOVE.par.value2 = MOVE.par.value2 + 0.1


	except Exception as inst:
		show_move.par.text = "NO HAND DETECTED"

		pass  

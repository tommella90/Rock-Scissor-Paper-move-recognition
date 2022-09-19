# me - this DAT
# scriptOp - the OP which is cooking
import random 

MOVE = op('gesture')
SCORE = op('score')

## random CPU move
def cpu_result():
    gestures = ['ROCK', 'PAPER', 'SCISSOR']
    cpu_result = random.choice(gestures)
    return cpu_result

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return

def onCook(scriptOp):

	## SCISSOR
	if MOVE.par.value0 > 0.9:
		op('button_reset').click()
		cpu_move = cpu_result()
		op('p_move').click()
		op('cpu_move').click()
		SCORE[1, "Player move"] = "SCISSOR"
		op('pause').click()

		if cpu_move == "SCISSOR":
			SCORE[1, "CPU move"] = "SCISSOR"
			op('EVEN').click()
			op('last_move').par.index = 2

		if cpu_move == "ROCK":
			SCORE[1, "CPU move"] = "ROCK"
			SCORE[1, "CPU score"] += 1
			op('cpu_win').click()
			op('last_move').par.index = 1

		if cpu_move == "PAPER":
			SCORE[1, "CPU move"] = "PAPER"
			SCORE[1, "Player score"] += 1
			op('player_win').click()
			op('last_move').par.index = 3

	## PAPER
	if MOVE.par.value1 > 0.9:
		op('button_reset').click()
		op('p_move').click()
		op('cpu_move').click()
		cpu_move = cpu_result()
		SCORE[1, "Player move"] = "PAPER"
		op('pause').click()

		if cpu_move == "PAPER":
			SCORE[1, "CPU move"] = "PAPER"
			op('EVEN').click()
			op('last_move').par.index = 3

		if cpu_move == "ROCK": 
			SCORE[1, "CPU move"] = "ROCK"
			SCORE[1, "Player score"] += 1
			op('player_win').click()
			op('last_move').par.index = 1

		if cpu_move == "SCISSOR": 
			SCORE[1, "CPU move"] = "SCISSOR"
			SCORE[1, "CPU score"] += 1
			op('cpu_win').click()
			op('last_move').par.index = 2

	## ROCK 
	if MOVE.par.value2 > 0.9:
		op('button_reset').click()
		cpu_move = cpu_result()
		op('p_move').click()
		op('cpu_move').click()
		SCORE[1, "Player move"] = "ROCK"
		op('pause').click()

		if cpu_move == "ROCK": 
			SCORE[1, "CPU move"] = "ROCK"
			op('EVEN').click()
			op('last_move').par.index = 1

		if cpu_move == "PAPER": 
			SCORE[1, "CPU move"] = "PAPER"
			SCORE[1, "CPU score"] += 1
			op('cpu_win').click()
			op('last_move').par.index = 3

		if cpu_move == "SCISSOR": 
			SCORE[1, "CPU move"] = "SCISSOR"
			SCORE[1, "Player score"] += 1
			op('player_win').click()
			op('last_move').par.index = 2


	## SHOW FINAL SCORE
	if SCORE[1, "Player score"] == 3:
		SCORE[1, "CPU score"] = 0 
		SCORE[1, "Player score"] = 0 
		SCORE[1, "Result"] = "YOU WON!!"
		op('win').click()

	if SCORE[1, "CPU score"] == 3:
		SCORE[1, "CPU score"] = 0 
		SCORE[1, "Player score"] = 0 
		SCORE[1, "Result"] = "LOOOOSER"
		op('lose').click()


	return

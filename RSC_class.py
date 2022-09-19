

class RockScissorPaper:

	RSC = op('hand_ml')
	SCORE = op('hand_ml/score')

	def __init__(self, myOp):
		self.myOp = myOp
		self.name = myOp.par.Name
		self.model = myOp.par.Model
		self.game_mode = myOp.par.Gamemode
		self.reset_game = myOp.par.Resetgame
		self.select_input = myOp.par.Inputdata

	## debug
		print(f"{self.name} is inizialised" )
		return


	def ChooseModel(self): 
		if self.model == 0:
			op('hand_ml/model_slector') == 0
			#op('hand_ml/show_model').par.text == "Model = Decision Tree"

		elif self.model == 1:
			op('hand_ml/model_slector') == 1
			#op('hand_ml/show_model').par.text == "Model = Knn"

		pass


	def ResetGame(self):
		op('hand_ml/reset_score').click()
		op('hand_ml/last_move').par.index = 0
		pass


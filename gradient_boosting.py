from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import GradientBoosting
gradient_boosting = GradientBoosting()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
gradient_boosting.fit(
	inputs=inputs,
	outputs=outputs,
	estimators=3, # equivalente ao número de árvores
	depth=5, # profundidade das árvores (número de níveis)
	minimum_samples_split=2, # número mínimo de alternativas para cada nó condicional da árvore
	learning_rate=1 # percentual da taxa de aprendizagem entre 0 (0%) e 1 (100%)
)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = gradient_boosting.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = gradient_boosting.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo
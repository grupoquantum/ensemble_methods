from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import Bagging
bagging = Bagging()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
bagging.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = bagging.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = bagging.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo
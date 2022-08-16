from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import DecisionTree
ensemble_tree = DecisionTree()

inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
ensemble_tree.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = ensemble_tree.predict(inputs=new_inputs, regression=True) # predição regressiva com o parâmetro regression=True
print(new_outputs) # como o algoritmo é focado em classificação os resultados não serão 100% precisos para casos lineares, porém poderão apresentar resultados satisfatórios em casos não lineares
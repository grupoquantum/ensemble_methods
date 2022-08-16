from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import DecisionTree
ensemble_tree = DecisionTree()

inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
ensemble_tree.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
new_outputs = ensemble_tree.predict(inputs=new_inputs)
print(new_outputs)
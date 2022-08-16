<div align="justify">
<p align="justify">
Os Métodos de Conjunto ou Ensemble Methods são algoritmos baseados em árvores de decisão que além da estrutura matemática construída com métodos estatísticos, também se aproveitam da lógica de programação com estruturas condicionais e laços de repetição aliados ao poder computacional do hardware para aplicarem diversas combinações desses algoritmos que trabalham em conjunto a fim conseguirem resultados melhores do que conseguiriam sozinhos.
</p>
<p align="justify">
A base dos Métodos de Conjunto está na arquitetura da Árvore de Decisão que se utiliza de uma estrutura em forma de árvore para organizar a lógica desses algoritmos. Confira a seguir um exemplo do algoritmo de Árvore de Decisão.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import DecisionTree
ensemble_tree = DecisionTree()

inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]
ensemble_tree.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
new_outputs = ensemble_tree.predict(inputs=new_inputs)
print(new_outputs)  
  </code>
</pre>
<br>
<pre>
  <code>
[['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas'], ['unidades'], ['dezenas'], ['centenas']]  
  </code>
</pre>
<br>
<p align="justify">
As Árvores de Decisão também podem emitir resultados regressivos mesmo tendo sido construídas primariamente com foco na classificação de dados categóricos. Confira abaixo um exemplo de Árvore de Decisão executando uma predição regressiva:
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import DecisionTree
ensemble_tree = DecisionTree()

inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
ensemble_tree.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
new_outputs = ensemble_tree.predict(inputs=new_inputs, regression=True) # predição regressiva com o parâmetro regression=True
print(new_outputs) # como o algoritmo é focado em classificação os resultados não serão 100% precisos para casos lineares, porém poderão apresentar resultados satisfatórios em casos não lineares  
  </code>
</pre>
<br>
<pre>
  <code>
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]  
  </code>
</pre>
<br>
<p align="justify">
Uma variação desse modelo é o Bagging (Empacotamento) que atua na diminuição da variância entre os dados com o objetivo de aumentar a generalização em resultados futuros. Isso poderá ser útil em casos onde as entradas da predição estiverem organizadas em uma distribuição muito diferente das entradas do treinamento.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]] 
  </code>
</pre>
<br>
<p align="justify">
Outro método de montagem que é bastante utilizado é o Bootstrap ou Inicialização Simulada que para aumentar a generalização cria amostras  simuladas de entradas baseadas nas entradas passadas para o treinamento, misturando dados reais a dados simulados.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import Bootstrap
bootstrap = Bootstrap()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
bootstrap.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = bootstrap.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = bootstrap.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]  
  </code>
</pre>
<br>
<p align="justify">
A Random Forest (Floresta Aleatória) talvez seja o mais conhecido dos algoritmos de montagem de conjunto, ele combina diversos algoritmos de Árvore de Decisão com dados escolhidos aleatoriamente.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import RandomForest
ensemble_forest = RandomForest()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
ensemble_forest.fit(inputs=inputs, outputs=outputs, number_of_trees=3) # number_of_trees define a quantidade de árvores na montagem do conjunto
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = ensemble_forest.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = ensemble_forest.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]
  </code>
</pre>
<br>
<p align="justify">
O Boosting ou algoritmo de Impulsionamento diminui os vieses de um conjunto de dados com a intenção de generalizar as saídas assim como alguns dos algoritmos anteriores, porém no Boosting isso é feito de maneira mais branda com o objetivo de se conseguir mais performance. Poderá ser útil em conjuntos muito grandes de dados onde o tempo de execução poderá ser penoso ao utilizarmos algoritmos mais complexos.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import Boosting
boosting = Boosting()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
boosting.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = boosting.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = boosting.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]  
  </code>
</pre>
<br>
<p align="justify">
O Gradient Boosting ou Impulsionamento Gradiente combina o algoritmo de Boosting a algoritmos de Árvore de Decisão onde uma árvore corrige os erros da outra de forma gradiente, ou seja, diminuindo o erro a cada nova árvore.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]  
  </code>
</pre>
<br>
<p align="justify">
O Adaptive Boosting ou AdaBoost consiste em uma adaptação do algoritmo de Boosting que tem como objetivo produzir resultados mais rápidos do que o Boosting mesmo que isso sacrifique parte da precisão nas repostas.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import AdaBoost
ada_boost = AdaBoost()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
ada_boost.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = ada_boost.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = ada_boost.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
[[4.2], [8.555555555555555], [12.692307692307693], [16.764705882352942], [20.80952380952381], [24.84], [28.862068965517242], [32.878787878787875], [36.891891891891895], [40.90243902439025]]  
  </code>
</pre>
<br>
<p align="justify">
O XGBoost (Extreme Gradient Boosting) ou Aumento Extremo de Gradiente é uma adaptação do Gradient Boosting que aplica um cálculo de gradiente descendente (erro mínimo) para diminuir a taxa de erro e usa o máximo do poder computacional para se obter o máximo de performance possível. Como seu foco está na performance os seus resultados poderão ser pouco precisos se comparados aos resultados dos algoritmos anteriores e isso se repete para os demais algoritmos daqui para baixo.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import XGBoost
xg_boost = XGBoost()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
xg_boost.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = xg_boost.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = xg_boost.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [23], [23], [27], [31], [35], [39]]
[[3], [5], [7], [9], [13], [13], [15], [17], [19], [21]]  
  </code>
</pre>
<br>
<p align="justify">
O LightGBM (Light Gradient Boosting Machine) ou Máquina de Aumento Leve de Gradiente é uma adaptação do algoritmo XGBoost feita com o objetivo de torná-lo ainda mais rápido. Para conseguir isso ele diminui a precisão no cálculo do gradiente descendente, o que poderá fazer com que ele tenha taxas de erro levemente superiores ao XGBoost porém com uma performance muito superior.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import LightGBM
light_gbm = LightGBM()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
light_gbm.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = light_gbm.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = light_gbm.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [23], [23], [27], [31], [35], [39]]
[[3.0], [5.0], [7.0], [9.0], [13.0], [13.0], [15.0], [17.0], [19.0], [21.0]]  
  </code>
</pre>
<br>
<p align="justify">
O CatBoost (Categorical Boosting) ou Impulsionamento Categórico separa os dados em categorias para reduzir o número de parâmetros e aumentar a velocidade de execução. Pelo fato de substituir os dados originais por categorias isso também faz com que ele diminua as possibilidades de overfitting (sobreajuste) que é quando o treinamento é concluído com taxas de erro muito baixas fazendo com que a predição perda a capacidade de generalizar para entradas diferentes das que foram utilizados no treinamento.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.ensemble_methods import CatBoost
cat_boost = CatBoost()
''' exemplo com classificação e regressão '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]]
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]]
cat_boost.fit(inputs=inputs, outputs=outputs)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]]
''' primeiro executa uma predição classificativa e depois uma predição regressiva '''
classification = cat_boost.predict(inputs=new_inputs, regression=False) # retorna saídas estáticas (uma das saídas do treinamento)
regression = cat_boost.predict(inputs=new_inputs, regression=True) # retorna saídas adaptativas
print(classification) # resultado classificativo
print(regression) # resultado regressivo  
  </code>
</pre>
<br>
<pre>
  <code>
[[3], [7], [11], [15], [23], [23], [27], [31], [35], [39]]
[[3.0], [5.0], [7.0], [9.0], [13.0], [13.0], [15.0], [17.0], [19.0], [21.0]]  
  </code>
</pre>
<br>
<p align="justify">
O desenvolvedor poderá testar cada um dos modelos para decidir qual deles aplicar. Alguns foram construídos com foco na precisão, outros com foco na velocidade e também há aqueles que ficam em um meio termo entre precisão e velocidade.
</*>
</div>

Environments:

Python 3.6
Tensor-Flow 1.6
Skleanr 0.20.1
numpy 1.15.0
csv

实验环境是用上述库搭建的，我和我的队友在这个环境均可以执行源代码。

1，首先需要将实验数据（diabetic_data.csv）放到项目根目录下；
2，运行两次getData.py进行数据预处理（one-hot编码），每次运行使用不同的attributes，getData.py文件中包括两组attribute数组，每次运行时讲另一份注释即可，（注意：操作后会产生大量数据）；
3，运行divideData.py来生成等分的十组数据（注意：操作后会产生大量数据）；
4，运行BranchNeuralNet.py，会自动训练十次模型，并计算相应的评价指标，并在最后取平均；
5，运行NaiveNeuralNet.py，会自动训练十次模型，并计算相应的评价指标，并在最后取平均；
6，运行DenseNeuralNet.py，会自动训练十次模型，并计算相应的评价指标，并在最后取平均；
7，运行completeDataSet.py,补全缺失数据；
8，运行DecisionTree.py,会自动预处理数据（label编码），同时随机等分十份数据，接下来会训练十个决策树模型，并计算相应评价指标，在最后会取平均。

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(" keys of iris_dataset:n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'])
# print("Target names:{}".format(iris_dataset['target_names']))
# print("feature names:{}".format(iris_dataset['feature_names']))
# print("Shape of data:{}".format(iris_dataset['data'].shape))
# print("target:{}".format(iris_dataset['target']))
# print("data:\n{}".format(iris_dataset['data']))

# import pandas as pd
# #直接读到pandas的数据框中
# data = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
# print(data)

# 对数据拆分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print("X_train:{}".format(X_train.shape))



# 观察数据，画图
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds ={'bins':20},s = 60,alpha=.8,cmap=mglearn.cm3)
plt.show()


# K近邻算法
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# 测试数据
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Prediction target name:{}".format(iris_dataset.target_names[prediction]))

print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))




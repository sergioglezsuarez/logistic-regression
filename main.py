import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from os import listdir

# Tarea 1
X, y = make_blobs(n_samples=[20, 20, 20], n_features=2, centers=np.array([[-7, -7], [-2, 10], [5, 2]]))
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)
perceptron = Perceptron().fit(Xtrain, ytrain)
logistic = LogisticRegression().fit(Xtrain, ytrain)

values1, values2, values3, values4, values5, values6 = ([], [], [], [], [], [])

for x in Xtrain:
    values1.append(((perceptron.coef_[0][0]*x[0])/perceptron.coef_[0][1])+(perceptron.intercept_[0]/perceptron.coef_[0][1]))
    values4.append(((logistic.coef_[0][0]*x[0])/logistic.coef_[0][1])+(logistic.intercept_[0]/logistic.coef_[0][1]))
    values2.append(((perceptron.coef_[1][0]*x[0])/perceptron.coef_[1][1])+(perceptron.intercept_[1]/perceptron.coef_[1][1]))
    values5.append(((logistic.coef_[1][0]*x[0])/logistic.coef_[1][1])+(logistic.intercept_[1]/logistic.coef_[1][1]))
    values3.append(((perceptron.coef_[2][0]*x[0])/perceptron.coef_[2][1])+(perceptron.intercept_[2]/perceptron.coef_[2][1]))
    values6.append(((logistic.coef_[2][0]*x[0])/logistic.coef_[2][1])+(logistic.intercept_[2]/logistic.coef_[2][1]))
    

fig, axs = plt.subplots()
axs.plot(pd.DataFrame(Xtrain)[0], values1, color="b", label='Superficie de decisión Perceptrón 1')
axs.plot(pd.DataFrame(Xtrain)[0], values2, color="b", label='Superficie de decisión Perceptrón 2')
axs.plot(pd.DataFrame(Xtrain)[0], values3, color="b", label='Superficie de decisión Perceptrón 3')
axs.plot(pd.DataFrame(Xtrain)[0], values4, color="r", label='Superficie de decisión Regresión Logística 1')
axs.plot(pd.DataFrame(Xtrain)[0], values5, color="r", label='Superficie de decisión Regresión Logística 2')
axs.plot(pd.DataFrame(Xtrain)[0], values6, color="r", label='Superficie de decisión Regresión Logística 3')
axs.scatter(pd.DataFrame(Xtrain)[0], pd.DataFrame(Xtrain)[1], color='black')
axs.legend()
plt.show()

preds1 = perceptron.predict(Xtest)
preds2 = logistic.predict(Xtest)
print(pd.DataFrame({"true": ytest, "perceptron": preds1, "logistic": preds2}))


# Tarea 2
train = pd.read_csv("CelebA-10K-train.csv")
Xtrain = train.iloc[:, 2:]
ytrain = train["Gender"]
test = pd.read_csv("CelebA-10K-test.csv")
Xtest = test.iloc[:, 2:]
ytest = test["Gender"]
modelo = LogisticRegression().fit(Xtrain, ytrain)
print("Test accuracy:", metrics.accuracy_score(modelo.predict(Xtest),ytest))

lista = listdir("ImagenesParaClasificar")

images = test[test["Image_name"].isin(lista)]

preds = modelo.predict(images.iloc[:, 2:])
print("Clasificación accuracy:", metrics.accuracy_score(preds, images["Gender"]))
print("Muestras mal clasificadas:")
for i in range(len(images)):
    if preds[i]!=images["Gender"].iloc[i]:
        print(images.iloc[i,0])
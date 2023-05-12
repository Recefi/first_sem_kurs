import matplotlib.pyplot as plt
import source.gen_selection as gs
import numpy as np
import source.function as fn
from sklearn import svm
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection

def runSVM(selection):
    X = selection[:,1:]
    #print(X)
    Y = selection[:,0]
    #print(Y)

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.20)
    # Основная часть
    clf = svm.SVC(kernel="linear")  # при C=1(по умолчанию) результаты почему-то лучше, чем при C=1000...
    clf.fit(X_train, Y_train)
    print(len(X_train))

    # Предсказание класса для X_test
    predictions = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(predictions, Y_test)*100)

    lambdas = clf.coef_[0].tolist()
    intercept = clf.intercept_[0]

    return lambdas, intercept


def drawSVM(selection, calcLam, machineLam, intercept, i, j):
    X = selection[:,1:]
    Y = selection[:,0]
    
    ax = plt.gca()
    ax.set(ylim=(-1, 1))
    ax.set_xlabel('M'+str(i+1))
    ax.set_ylabel('M'+str(j+1))
    ax.scatter(X[:, i], X[:, j], c=Y, s=1, cmap=plt.cm.Paired)

    x_visual = np.linspace(-1,1)
    y_visual = -(calcLam[i] / calcLam[j]) * x_visual # - intercept / calcLam[j]
    ax.plot(x_visual, y_visual, color="red", label="Calc")
    y_visual = -(machineLam[i] / machineLam[j]) * x_visual - intercept / machineLam[j]
    ax.plot(x_visual, y_visual, color="blue", label="SVM")
    ax.legend()

    plt.draw()


def showAllSVM(selection, calcLam, machineLam, intercept):
    X = selection[:,1:]
    Y = selection[:,0]

    fig, ax = plt.subplots(8,8)
    for i in range(8):
        for j in range(8):
            if i!=j :
                ax[i][j].set(ylim=(-1, 1))
                # ax[i][j].set_xlabel('M'+str(i+1))
                # ax[i][j].set_ylabel('M'+str(j+1))
                ax[i][j].scatter(X[:, i], X[:, j], c=Y, s=1, cmap=plt.cm.Paired)
                x_visual = np.linspace(-1, 1)
                y_visual = -(calcLam[i] / calcLam[j]) * x_visual # - intercept / calcLam[j]
                ax[i][j].plot(x_visual, y_visual, color="red")
                y_visual = -(machineLam[i] / machineLam[j]) * x_visual - intercept / machineLam[j]
                ax[i][j].plot(x_visual, y_visual, color="blue")

    plt.show()

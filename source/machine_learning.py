import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def runSVM(selection):
    X = selection[:,1:]
    #print(X)
    Y = selection[:,0]
    #print(Y)

    print(len(X))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
    # Основная часть
    clf = svm.SVC(kernel="linear", C=1000)  # C=1000 для данной выборки обязательно!!! иначе по восстановленным коэф-там (+,-) побеждает (-,-), если выборка другая и получаются плохие результаты, то имеет смысл порегулировать...
    clf.fit(X_train, Y_train)

    # Предсказание класса для X_test
    predictions = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(Y_test, predictions)*100)

    lambdas = clf.coef_[0]
    intercept = clf.intercept_[0]

    return lambdas, intercept


def drawSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept, i, j):
    X = selection[:,1:]
    Y = selection[:,0]
    
    fig, ax = plt.subplots()
    ax.set(ylim=(-1, 1))
    ax.set_xlabel('M'+str(i+1))
    ax.set_ylabel('M'+str(j+1))
    ax.scatter(X[:, i], X[:, j], c=Y, s=5, cmap=plt.cm.Paired)

    x_visual = np.linspace(-1,1)
    y_visual = -(norm_calcCoefs_mf[i] / norm_calcCoefs_mf[j]) * x_visual # - intercept / norm_calcCoefs[j]  # хз как подсчитать intercept для вычисляемых коэф-тов... но это и не важно 
    ax.plot(x_visual, y_visual, color="red", label="Taylor_maxFitPnt")
    y_visual = -(norm_calcCoefs_n[i] / norm_calcCoefs_n[j]) * x_visual # - intercept / norm_calcCoefs[j]  # мб использовать пропущенные слагаемые из полной формулы Тейлора...
    ax.plot(x_visual, y_visual, color="green", label="Taylor_nearPnt")
    y_visual = -(norm_machCoefs[i] / norm_machCoefs[j]) * x_visual - intercept / norm_machCoefs[j]
    ax.plot(x_visual, y_visual, color="blue", label="SVM")
    ax.legend()

    plt.draw()


def showAllSVM(selection, norm_machCoefs, norm_calcCoefs_mf, norm_calcCoefs_n, intercept):
    X = selection[:,1:]
    Y = selection[:,0]

    fig, ax = plt.subplots(8,8)
    for i in range(8):
        for j in range(8):
            if i!=j :
                ax[i][j].set(ylim=(-1, 1))
                # ax[i][j].set_xlabel('M'+str(i+1))
                # ax[i][j].set_ylabel('M'+str(j+1))
                ax[i][j].scatter(X[:, i], X[:, j], c=Y, s=5, cmap=plt.cm.Paired)
                x_visual = np.linspace(-1, 1)
                y_visual = -(norm_calcCoefs_mf[i] / norm_calcCoefs_mf[j]) * x_visual # - intercept / norm_calcCoefs[j]
                ax[i][j].plot(x_visual, y_visual, color="red")
                y_visual = -(norm_calcCoefs_n[i] / norm_calcCoefs_n[j]) * x_visual # - intercept / norm_calcCoefs[j]
                ax[i][j].plot(x_visual, y_visual, color="green")
                y_visual = -(norm_machCoefs[i] / norm_machCoefs[j]) * x_visual - intercept / norm_machCoefs[j]
                ax[i][j].plot(x_visual, y_visual, color="blue")

    plt.show()

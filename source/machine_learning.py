import matplotlib.pyplot as plt
import gen_selection as gs
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

# we create 40 separable points
X = []
y = []
for v in range(len(gs.norm_Macroparameters[1])):
    for lambdas in range(v,len(gs.norm_Macroparameters[1])):
        x=[]
        
        if (gs.classification_Table[v][lambdas]==1):
            for i in range(8):
                x.append(gs.norm_Macroparameters[i][v]-gs.norm_Macroparameters[i][lambdas])
            for i in range(8):
                for j in range(i,8):
                    x.append(gs.M_sqr[i][j][v]-gs.M_sqr[i][j][lambdas])
            y.append(gs.classification_Table[v][lambdas])
            X.append(x)
        if (gs.classification_Table[v][lambdas]==-1):
            for i in range(8):
                x.append(gs.norm_Macroparameters[i][lambdas]-gs.norm_Macroparameters[i][v])
            for i in range(8):
                for j in range(i,8):
                    x.append(gs.M_sqr[i][j][lambdas]-gs.M_sqr[i][j][v])
            y.append(gs.classification_Table[v][lambdas])
            X.append(x)
                    
X= np.array(X)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 5], c=y, s=3, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
# DecisionBoundaryDisplay.from_estimator(
#     clf,
#     X,
#     plot_method="contour",
#     colors="k",
#     levels=[-1, 0, 1],
#     alpha=0.5,
#     linestyles=["--", "-", "--"],
#     ax=ax,
# )
# plot support vectors
lambdas = clf.coef_[0]
b = clf.intercept_[0]
x_visual = np.linspace(-1,1)
y_visual = -(lambdas[0] / lambdas[5]) * x_visual - b / lambdas[5]
ax.plot(x_visual, y_visual)
# ax.scatter(
#     clf.support_vectors_[:, 0],
#     clf.support_vectors_[:, 5],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
ax.scatter(X[:, 0], X[:, 5], c=y, s=3, cmap=plt.cm.Paired)
plt.show()
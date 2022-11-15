import matplotlib.pyplot as plt
import gen_selection as gs
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

# we create 40 separable points
X = []
y = []
for v in range(len(gs.norm_Macroparameters)):
    for w in range(v,len(gs.norm_Macroparameters)):
        x=[]
        
        if (gs.classification_Table[v][w]==1):
            for i in range(44):
                x.append(gs.norm_Macroparameters[i][v]-gs.norm_Macroparameters[i][w])
            y.append(gs.classification_Table[v][w])
            X.append(x)
        if (gs.classification_Table[v][w]==-1):
            for i in range(44):
                x.append(gs.norm_Macroparameters[i][w]-gs.norm_Macroparameters[i][v])
            y.append(gs.classification_Table[v][w])
            X.append(x)
                    

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()
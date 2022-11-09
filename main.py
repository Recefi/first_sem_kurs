import numpy as np
import random 
from cmath import pi
# from plotly import graph_objs as go
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt



_D = 100
sigma1 = 0.25
sigma2 = 0.003
alpha_j=0.0016
alpha_a=0.006
beta_j=0.0000007
beta_a=0.000000075
gamma_j=0.00008
gamma_a=0.004
delta_j=0.000016
delta_a=0.00006

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


Selections = [[] for i in range(4)]

n=300
#Генервция А и В:
for i in range(n): 

    __a_adult = random.random()*(-_D)
    __b_adult = random.random()*(min(-__a_adult, __a_adult+_D))
    __a_jun = random.random()*(-_D)
    __b_jun = random.random()*(min(-__a_jun, __a_jun+_D))
    
    for j in range(-1,2,2):
        
        Selections[0].append(__a_adult)
        Selections[1].append(__b_adult*j)


        Selections[2].append(__a_jun)
        Selections[3].append(__b_jun*j)

        

for i in Selections:
    print(i,'\n')

J = []

MPs = [[] for i in range(8)]
for i in range(2*n):
    for z in [-1,1]:
        MPs[0].append(z*sigma1*(Selections[0][i] + _D))
        MPs[1].append(z*-sigma2*(Selections[0][i] + _D + Selections[1][i]/2))
        MPs[2].append(z*-2*pi*pi*Selections[1][i]*Selections[1][i])
        MPs[3].append(z*-(Selections[0][i]+_D/2)*(Selections[0][i]+_D/2)+Selections[1][i]*Selections[1][i]/2)
        MPs[4].append(z*sigma1*(Selections[2][i]+_D))
        MPs[5].append(z*-sigma2*(Selections[0][i]+_D+Selections[3][i]/2))
        MPs[6].append(z*-2*pi*pi*Selections[1][i]*Selections[3][i])
        MPs[7].append(z*-(Selections[2][i]+_D/2)*(Selections[2][i]+_D/2)+Selections[3][i]*Selections[3][i]/2)

ind=[]
for i in range(len(MPs[1])):
    p = alpha_j*MPs[0][i]+beta_j*MPs[2][i]+delta_j*MPs[3][i]
    r = alpha_a*MPs[4][i]+beta_a*MPs[6][i]+delta_a*MPs[7][i]
    q = gamma_j*MPs[1][i]
    s = gamma_a*MPs[5][i]
    if ((4*r*p+np.square(p+q+s))<0):
        ind.append(i)
    else:
        J.append(-s-p+q+(np.sqrt((4*r*p+np.square(p+q+s)))))
k=0
for i in ind:
    k-=1
    for j in range(8):
        del MPs[j][i+k]



def NormMP(MPs):
    normMP=[]
    mMax=max([abs(mp) for mp in MPs])
    for mp in MPs:
        normMP.append(mp/mMax)
    return normMP
normMPs=[NormMP(mp) for mp in MPs]








#plt.style.use(' plt-gallery')
t= np.linspace(0,1,100)

x= Selections[0][0]+Selections[0][1]*np.cos(2*pi*t)
x2= Selections[0][2]+Selections[0][3]*np.cos(2*pi*t)
xm=min(x+x2)
fig, ax = plt.subplots()

ax.plot(t, x, linewidth=2.0)
ax.plot(t, x2, linewidth=2.0)

ax.set(xlim=(0, 1), xticks=np.arange(0, 1),
       ylim=(xm, 0), yticks=np.arange(-1, 0))
plt.show()

fig, histMp = plt.subplots(2, 4)
for i in range(2):
    for j in range(4):
        histMp[i][j].hist(normMPs[i*4+j], bins=50, linewidth=0.5, edgecolor="white")
        histMp[i][j].set(xlim=(-1, 1), xticks=np.linspace(-1, 1, 9),
                         ylim=(0, 100), yticks=np.linspace(0, 100, 9))
        histMp[i][j].set_title("normM{:X}".format(i*4+j+1))
plt.show()

corM=np.corrcoef(MPs)

fig, cor = plt.subplots()
im = cor.imshow(corM)

cor.set_xticks(np.arange(8), labels=["M{:X}".format(i+1) for i in range(8)])
cor.set_yticks(np.arange(8), labels=["M{:X}".format(i+1) for i in range(8)])



for i in range(8):
    for j in range(8):
        text = cor.text(j, i, corM[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()


fig, ax = plt.subplots()

im, cbar = heatmap(corM, ["M{:X}".format(i+1) for i in range(8)], ["M{:X}".format(i+1) for i in range(8)], ax=ax,
                   cmap="RdYlBu_r", cbarlabel=" ")
texts = annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.show()
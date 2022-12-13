import matplotlib.pyplot as plt
import numpy as np
import source.gen_selection as gs

def get_table(title,xNames,yNames,array):
    fig, ax = plt.subplots() 
    ax.set_axis_off() 

    table = ax.table( 
        cellText = array,    
        rowLabels = yNames,
        colLabels = xNames,
        cellLoc ='center',  
        loc ='upper left')         

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    ax.set_title(title,fontweight ="bold") 

    plt.show() 


def get_sinss():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1)
    y = gs.A_jun[gs.maxf_ind] + gs.B_jun[gs.maxf_ind] * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="red")
    x = np.linspace(0, 1)
    y = gs.A_adult[gs.maxf_ind] + gs.B_adult[gs.maxf_ind] * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="blue")
    plt.show

def get_gistogram(array, tittle):
    a_min = min(array)
    a_max = max(array)
    fig, histMp = plt.subplots()

    histMp.hist(array, bins=50, linewidth=0.5, edgecolor="white")

    histMp.set(xlim=(a_min, a_max), xticks=np.linspace(a_min, a_max, 9))
    histMp.set_title(tittle)
    plt.show()




def get_correllation(array, arg_names):
    array_cor=np.round(np.corrcoef(array),2)
    
    fig, cor = plt.subplots()
    im = cor.imshow(array_cor)

    cor.set_xticks(np.arange(8), labels=arg_names)
    cor.set_yticks(np.arange(8), labels=arg_names)

    for i in range(8):
        for j in range(8):
            text = cor.text(j, i, array_cor[i, j],
                        ha="center", va="center", color="r")

    fig.tight_layout()
    plt.show()
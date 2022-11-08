import numpy as np
import random 
from cmath import pi
from plotly import graph_objs as go
from dash import Dash, html, Input, Output, dash_table
import pandas as pd
import plotly.graph_objects as go
import matplotlib as mpl

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



Selections = [[] for i in range(4)]

n=10
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

MPs = [[] for i in range(8)]
for i in range(2*n):
    MPs[0].append(sigma1*(Selections[0][i] + _D))
    MPs[1].append(-sigma2*(Selections[0][i] + _D + Selections[1][i]/2))
    MPs[2].append(-2*pi*pi*Selections[1][i]*Selections[1][i])
    MPs[3].append(-(Selections[0][i]+_D/2)*(Selections[0][i]+_D/2)+Selections[1][i]*Selections[1][i]/2)
    MPs[4].append(sigma1*(Selections[2][i]+_D))
    MPs[5].append(-sigma2*(Selections[0][i]+_D+Selections[3][i]/2))
    MPs[6].append(-2*pi*pi*Selections[1][i]*Selections[3][i])
    MPs[7].append(-(Selections[2][i]+_D/2)*(Selections[2][i]+_D/2)+Selections[3][i]*Selections[3][i]/2)



def NormMP(MPs):
    normMP=[]
    mMax=max([abs(mp) for mp in MPs])
    for mp in MPs:
        normMP.append(mp/mMax)
    return normMP
normMPs=[NormMP(mp) for mp in MPs]

J = []

for i in range(2*n):
    p = alpha_j*MPs[0][i]+beta_j*MPs[2][i]+delta_j*MPs[3][i]
    r = alpha_a*MPs[4][i]+beta_a*MPs[6][i]+delta_a*MPs[7][i]
    q = gamma_j*MPs[1][i]
    s = gamma_a*MPs[5][i]
    J.append(-s-p+q+(np.sqrt((4*r*p+np.square(p+q+s)))))






fig = go.Figure(data=[go.Table(header=dict(values=['A adult ', 'B adult','A jun', 'B jun',]),
cells=dict(values=[Selections[i]for i in range(4)]))
                     ])
fig.show()

MPfig = go.Figure(data=[go.Table(header=dict(values=["M{:X}".format(i+1) for i in range(8)]),
cells=dict(values=[MPs[i]for i in range(8)]))
                     ])
MPfig.show()

NMPfig = go.Figure(data=[go.Table(header=dict(values=["normM{:X}".format(i+1) for i in range(8)]),
cells=dict(values=[normMPs[i]for i in range(8)]))
                     ])
NMPfig.show()

JPfig = go.Figure(data=[go.Table(header=dict(values=["J"]),
cells=dict(values=[J]))
                     ])
JPfig.show()

#mpl.style.use('_mpl-gallery')

x= Selections[0][0]+Selections[0][1]*np.cos(t)
t= np.linspace(0,0.2,0.4,0.6,0.8,1)

fig, ax = mpl.subplots()

ax.plot(t, x, linewidth=2.0)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

mpl.show()
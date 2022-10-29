from cmath import pi
from xmlrpc.client import MAXINT
import numpy as np
import random
import math
_D = 100


Selections = []



for i in range(10): 

    __a_adult = random.random()*(-_D)
    __b_adult = random.random()*(min(-__a_adult, __a_adult+_D))
    __a_jun = random.random()*(-_D)
    __b_jun = random.random()*(min(-__a_jun, __a_jun+_D))
    
    for j in range(-1,2,2):
        __hlist = []
        __hlist.append(__a_adult)
        __hlist.append(__b_adult*j)


        __hlist.append(__a_jun)
        __hlist.append(__b_jun*j)

        Selections.append(__hlist)

for i in Selections:
    print(i,'\n')

MPs = []

sigma1 = 0.25
sigma2 = 0.003

for Selection in Selections:
    MP = []
    MP.append(sigma1*(Selection[0] + _D))
    MP.append(-sigma1*(Selection[0] + _D + Selection[1]/2))
    MP.append(-2*pi*pi*Selection[1]*Selection[1]/2)
    MP.append(-(Selection[0]+_D/2)*(Selection[0]+_D/2)+Selection[1]/2)
    MP.append(sigma1*(Selection[2]+_D))
    MP.append(-sigma2*(Selection[0]+_D+Selection[3]/2))
    MP.append(-2*pi*pi*Selection[1]*Selection[3]/2)
    MP.append(-(Selection[2]+_D/2)*(Selection[2]+_D/2)+Selection[3]/2)

    MPs.append(MP)

for i in MPs:
    print(i,'\n')

alpha_j=0.0016
alpha_a=0.006
beta_j=0.0000007
beta_a=0.000000075
gamma_j=0.00008
gamma_a=0.004
delta_j=0.000016
delta_a=0.00006

J = []

for MP in MPs:
    p = alpha_j*MP[0]+beta_j*MP[2]+delta_j*MP[3]
    r = alpha_a*MP[4]+beta_a*MP[6]+delta_a*MP[7]
    q = gamma_j*MP[1]
    s = gamma_a*MP[5]
    J.append(-s-p+q+(np.sqrt(np.abs(4*r*p+np.square(p+q+s)))))

    

for j in J:
    print(j)

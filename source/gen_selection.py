import random
import source.param as param
import math
import numpy as np

#Двустадийная модель
#А - отвечает за глубину погружения
#B - за амплитуду колебаний
#Количество параметров А и В
n = 100 #//4
A_jun = []
B_jun = []
A_adult = []
B_adult = []
for i in range(n//4):
    for sign in [-1,1]:
        for sign_2 in [-1,1]:
            A_jun.append(random.random()*(-param.depth))
            B_jun.append(random.random()* min(-A_jun[i], A_jun[i]+param.depth)*sign)

            A_adult.append(random.random()*(-param.depth))
            B_adult.append(random.random()* min(-A_adult[i], A_adult[i]+param.depth)*sign_2)



Macroparameters = [[] for i in range(8)]
for i in range(n):
    Macroparameters[0].append(param.sigma1 * (A_jun[i] + param.depth))
    Macroparameters[1].append(-param.sigma2*(A_jun[i] + param.depth + B_jun[i]/2))
    Macroparameters[2].append(-2*np.square(math.pi*B_jun[i]))
    Macroparameters[3].append(-(np.square(A_jun[i]+param.optimal_depth)-np.square(B_jun[i])/2))
    Macroparameters[4].append(param.sigma1*(A_adult[i] + param.depth))
    Macroparameters[5].append(-param.sigma2*(A_adult[i] + param.depth + B_adult[i]/2))
    Macroparameters[6].append(-2*np.square(math.pi*B_adult[i]))
    Macroparameters[7].append(-(np.square(A_adult[i]+param.optimal_depth)-np.square(B_adult[i])/2))



Fitness = []
index_of_bad_Mps = []
for i in range(n):
    p = param.alpha_j*Macroparameters[0][i] + param.beta_j*Macroparameters[2][i] + param.delta_j*Macroparameters[3][i]
    r = param.alpha_a*Macroparameters[4][i] + param.beta_a*Macroparameters[6][i] + param.delta_a*Macroparameters[7][i]
    q = param.gamma_j*Macroparameters[1][i]
    s = param.gamma_a*Macroparameters[5][i]

    if((4*r*p+np.square(p+q+s))<0):
        index_of_bad_Mps.append(i)
    else:
        Fitness.append(-s-p-q+(np.sqrt((4*r*p+np.square(p+q+s)))))

#Очистка
k=0
for i in range(len(index_of_bad_Mps)):
    for j in range(8):
        Macroparameters[j].pop(i)
    k-=1

        


# M_diff = [[[]for i in range(len(Macroparameters[1]))] for j in range(len(Macroparameters[1]))]

# for v in range(len(Macroparameters[1])):
#     for w in range(len(Macroparameters[1])):
#         for i in range(8):
#             M_diff[v][w].append(Macroparameters[i][v]-Macroparameters[i][w])


M_sqr = [[[]for i in range(8)] for j in range(8)]
for i in range(8):
    for j in range(8):
        for k in range(len(Macroparameters[1])):
            M_sqr[j][i].append(Macroparameters[i][k]*Macroparameters[j][k])


# M_sqr_diff = [[[[]for i in range(len(Macroparameters[1]))]for j in range(8)]for k in range(8)]


# for v in range(len(Macroparameters[1])):
#     for w in range(len(Macroparameters[1])):
#         for i in range(8):
#             for j in range(8):
#               M_sqr_diff[i][j][v].append(M_sqr[i][j][v]-M_sqr[i][j][w])

# M_diff_norm=[[[]for i in range(len(Macroparameters[1]))] for j in range(len(Macroparameters[1]))]
# for i in range(8):
#     M_max=max([(abs(m[i])for m in mp) for mp in M_diff])
#     for v in range(len(Macroparameters[1])):
#         for w in range(len(Macroparameters[1])):
#             M_diff[v][w].append(M_diff[v][w][i]/M_max)

# M_sqr_diff_norm=[[[[]for i in range(len(Macroparameters[1]))]for j in range(len(Macroparameters[1]))]for k in range(8)]

# for i in range(8):
#     for j in range(8):
#         M_max=max([(abs(m[i][j])for m in mp) for mp in M_diff])
#         for v in range(len(Macroparameters[1])):
#             for w in range(len(Macroparameters[1])):
#                 M_sqr_diff_norm[v][w][i].append(M_sqr_diff[v][w][i][j]/M_max)


# norm_Macroparameters = [[] for i in range(8)]
# for i in range(8):
#     max_param = max([abs(mp) for mp in Macroparameters[i]])
#     for j in range(len(Macroparameters[i])):
#         norm_Macroparameters[i].append(Macroparameters[i][j]/max_param)









classification_Table = np.zeros((len(Fitness),len(Fitness)))
for i in range(len(Fitness)):
    for j in range(i,len(Fitness)):
        if i == j:
            classification_Table[i,j] = 0
            continue  
        if Fitness[i]>Fitness[j]:
            classification_Table[i,j] = 1
            classification_Table[j,i] = -1
        else:
            classification_Table[i,j] = -1
            classification_Table[j,i] = 1


import random
import source.param as param
import math
import numpy as np
import pandas as pd

#Двустадийная модель
#А - отвечает за глубину погружения
#B - за амплитуду колебаний
#Количество параметров А и В
n = 100 #//4
A_jun = []
B_jun = []
A_adult = []
B_adult = []
for i in range(n):
    a_j = np.random.random()*(-param.depth)
    m_j = min(-a_j, a_j+param.depth)
    b_j = np.random.uniform(-m_j, m_j)
    a_a = np.random.random()*(-param.depth)
    m_a = min(-a_a, a_a+param.depth)
    b_a = np.random.uniform(-m_a, m_a)

    A_jun.append(a_j)
    A_jun.append(a_j)
    A_adult.append(a_a)
    A_adult.append(a_a)
    B_jun.append(b_j)
    B_jun.append(-b_j)
    B_adult.append(b_a)
    B_adult.append(-b_a)

# A_jun.append(-20.73)
# B_jun.append(-3.93)
# A_adult.append(-51.10)
# B_adult.append(-39.17)


# data1 =pd.DataFrame(data = {'Aj': A_jun, 'Bj': B_jun})
# data2 =pd.DataFrame(data = {'Aa': A_adult, 'Ba': B_adult})
# data=pd.concat([data1, data2], axis=1)
# # print(data)
# data.to_csv("data.csv", index=True)

stratData = pd.read_csv("data.csv")
A_jun = stratData['Aj'].tolist()
A_adult = stratData['Aa'].tolist() 
B_jun = stratData['Bj'].tolist()
B_adult = stratData['Ba'].tolist() 
n = stratData.shape[0] // 2

maxf=0
maxf_ind=0
k = 0
Fitness = []
for i in range(2*n):
    res = []

    M1 = param.sigma1 * (A_jun[i] + param.depth)
    M2 = -param.sigma2*(A_jun[i] + param.depth + B_jun[i]/2)
    M3 = -2*np.square(math.pi*B_jun[i])
    M4 = -(np.square(A_jun[i]+param.optimal_depth)-np.square(B_jun[i])/2)
    M5 = param.sigma1*(A_adult[i] + param.depth)
    M6 = -param.sigma2*(A_adult[i] + param.depth + B_adult[i]/2)
    M7 = -2*np.square(math.pi*B_adult[i])
    M8 = -(np.square(A_adult[i]+param.optimal_depth)-np.square(B_adult[i])/2)

    p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
    r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
    q = param.gamma_j*M2
    s = param.gamma_a*M6
    if(4*r*p+np.square(p+q-s)>=0):
        fit = -s-p-q+(np.sqrt((4*r*p+np.square(p+q-s))))
        print('fit',fit)
        if fit>maxf:
            maxf=fit
            maxf_ind=i    

        res = [fit,M1,M2,M3,M4,M5,M6,M7,M8]
        print(p)
        k+=1
        for m in range(8):
            for j in range(m,8):
                res.append(res[m+1]*res[j+1])
        Fitness.append(res)

print("maxf:",maxf)
print("A max", A_jun[maxf_ind])
print(k)

MColumns = ['fit']
for i in range(1, 9):
    MColumns.append('M' + str(i))
for i in range(1, 9):
    for j in range(i, 9):
        MColumns.append('M'+str(i) + 'M'+str(j))

fit_data =pd.DataFrame(Fitness, columns=MColumns)
#print(fit_data)
fit_data.to_csv("fit_data.csv", index=False)


classification_Table = np.zeros((len(Fitness),len(Fitness)))

for i in range(len(Fitness)):
    for j in range(i,len(Fitness)):
        if i == j:
            classification_Table[i,j] = 0
            continue  
        if Fitness[i][0]>Fitness[j][0]:
            classification_Table[i,j] = 1
            classification_Table[j,i] = -1
        else:
            classification_Table[i,j] = -1
            classification_Table[j,i] = 1

selection = []


for i in range(len(Fitness)):
    for j in range(len(Fitness)):
        if(j==i):
            continue
        res = []
        res.append(classification_Table[i][j])
        for k in range(1,len(Fitness[i])):
            res.append(Fitness[i][k]-Fitness[j][k])
        selection.append(res)

selection = np.array(selection)

MColumns[0] = 'sel'
sel_data = pd.DataFrame(selection, columns=MColumns)
#print(sel_data)
sel_data.to_csv("sel_data.csv", index=False)

# Нормирование
for i in range(len(selection[0])):
    max = np.max(np.abs(selection[:,i]))
    selection[:,i]/=max

norm_sel_data = pd.DataFrame(selection, columns=MColumns)
#print(norm_sel_data)
norm_sel_data.to_csv("norm_sel_data.csv", index=False)

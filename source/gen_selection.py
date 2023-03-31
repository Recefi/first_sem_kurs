import random
import source.param as param
import math
import numpy as np
import pandas as pd

#Двустадийная модель
#А - отвечает за среднюю глубину погружения
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

# # Лучшая точка у других групп
# A_jun.append(-20.73)
# B_jun.append(-3.93)
# A_adult.append(-51.10)
# B_adult.append(-39.17)

# # Заносим A и B в файл
# stratData1 = pd.DataFrame(data = {'Aj': A_jun, 'Bj': B_jun})
# stratData2 = pd.DataFrame(data = {'Aa': A_adult, 'Ba': B_adult})
# stratData = pd.concat([stratData1, stratData2], axis=1)
# # print(stratData)
# stratData.to_csv("data.csv", index=True)

# Считываем A и B из файла
strat_data = pd.read_csv("data.csv")
A_jun = strat_data['Aj'].tolist()
A_adult = strat_data['Aa'].tolist() 
B_jun = strat_data['Bj'].tolist()
B_adult = strat_data['Ba'].tolist() 

maxf=0
maxf_ind=0
k = 0
Fitness = []
Indexes = []
for i in range(strat_data.shape[0]):  # 2*n, если вручную не редактируем выборку
    res = []

    M1 = param.sigma1 * (A_jun[i] + param.depth)
    M2 = -param.sigma2 * (A_jun[i] + param.depth + B_jun[i]/2)
    M3 = -2*(math.pi*B_jun[i])**2
    M4 = -((A_jun[i]+param.optimal_depth)**2-(B_jun[i]**2)/2)
    M5 = param.sigma1 * (A_adult[i] + param.depth)
    M6 = -param.sigma2 * (A_adult[i] + param.depth + B_adult[i]/2)
    M7 = -2*(math.pi*B_adult[i])**2
    M8 = -((A_adult[i]+param.optimal_depth)**2-(B_adult[i]**2)/2)

    p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
    r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
    q = param.gamma_j*M2
    s = param.gamma_a*M6
    if(4*r*p+np.square(p+q-s)>=0):
        fit = -s-p-q+(np.sqrt((4*r*p+(p+q-s)**2)))
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
        Indexes.append(i)

print("maxf:",maxf)
print("maxf_ind",maxf_ind)
print(k)

MColumns = ['fit']
for i in range(1, 9):
    MColumns.append('M' + str(i))
for i in range(1, 9):
    for j in range(i, 9):
        MColumns.append('M'+str(i) + 'M'+str(j))

# # Заносим Fitness в файл
# fit_data = pd.DataFrame(Fitness, columns=MColumns)
# fit_data.index = Indexes
# #print(fit_data)
# fit_data.to_csv("fit_data.csv", index=True)

# Считываем Fitness из файла
fit_data = pd.read_csv("fit_data.csv", index_col=0)
Fitness = fit_data.values.tolist()
# Получаем индекс макс.значения фитнеса с учетом редактирования fit_data.csv
maxf_ind = fit_data[['fit']].idxmax(axis='index')[0]
print("new maxf_ind",maxf_ind)

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
# Заносим selection в файл
sel_data = pd.DataFrame(selection, columns=MColumns)
#print(sel_data)
sel_data.to_csv("sel_data.csv", index=False)

# Нормирование
for i in range(len(selection[0])):
    max = np.max(np.abs(selection[:,i]))
    selection[:,i]/=max

# Заносим нормированный selection в файл
norm_sel_data = pd.DataFrame(selection, columns=MColumns)
#print(norm_sel_data)
norm_sel_data.to_csv("norm_sel_data.csv", index=False)

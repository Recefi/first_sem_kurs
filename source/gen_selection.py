import source.param as param
import numpy as np
import pandas as pd

def genStrats(n):
    """Генерация стратегий"""
    A_jun = []
    B_jun = []
    A_adult = []
    B_adult = []
    for i in range(n):
        a_j = np.random.random()*(-param.depth)
        m_j = min(-a_j, a_j+param.depth)
        b_j = np.random.uniform(0, m_j)
        a_a = np.random.random()*(-param.depth)
        m_a = min(-a_a, a_a+param.depth)
        b_a = np.random.uniform(0, m_a)

        A_jun.append(a_j)
        B_jun.append(b_j)
        A_adult.append(a_a)
        B_adult.append(b_a)

        A_jun.append(a_j)
        B_jun.append(-b_j)
        A_adult.append(a_a)
        B_adult.append(b_a)

        A_jun.append(a_j)
        B_jun.append(b_j)
        A_adult.append(a_a)
        B_adult.append(-b_a)

        A_jun.append(a_j)
        B_jun.append(-b_j)
        A_adult.append(a_a)
        B_adult.append(-b_a)

    return A_jun, B_jun, A_adult, B_adult

def calcFitness(stratData):
    """
    Подсчет фитнеса и макропараметров
        Возвращает: Fitness, FitIndxs, maxf_ind
            FitIndxs[индекс Fitness] = исходный индекс
                pqrsData и maxf_ind в исходных индексах, а не в индексах Fitness
    """
    A_jun = stratData['Aj']
    B_jun = stratData['Bj']
    A_adult = stratData['Aa']
    B_adult = stratData['Ba']

    maxf=0
    maxf_ind=0
    k = 0
    Fitness = []
    FitIndxs = []
    pqrs = []
    for i in A_jun.index:  # используем исходные индексы
        res = []

        M1 = param.sigma1 * (A_jun[i] + param.depth)
        M2 = -param.sigma2 * (A_jun[i] + param.depth + B_jun[i]/2)
        M3 = -2*(np.pi*B_jun[i])**2
        M4 = -((A_jun[i]+param.optimal_depth)**2 + (B_jun[i]**2)/2)

        M5 = param.sigma1 * (A_adult[i] + param.depth)
        M6 = -param.sigma2 * (A_adult[i] + param.depth + B_adult[i]/2)
        M7 = -2*(np.pi*B_adult[i])**2
        M8 = -((A_adult[i]+param.optimal_depth)**2 + (B_adult[i]**2)/2)

        p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
        r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
        q = -param.gamma_j*M2
        s = -param.gamma_a*M6

        # print("p = ", p)
        # print("r = ", r)
        # print("q = ", q)
        # print("s = ", s)

        if(4*r*p+np.square(p+q-s)>=0):
            fit = -s-p-q+(np.sqrt((4*r*p+(p+q-s)**2)))
            print('fit',fit)
            if fit>maxf:
                maxf=fit
                maxf_ind=i

            res = [fit,M1,M2,M3,M4,M5,M6,M7,M8]

            k+=1
            for m in range(8):
                for j in range(m,8):
                    res.append(res[m+1]*res[j+1])
            Fitness.append(res)
            FitIndxs.append(i)
            pqrs.append([p, q, r, s])
    
    print("init_maxf:",maxf)
    print("init_maxf_ind:",maxf_ind)
    print("strats:",k)

    pqrsData = pd.DataFrame(pqrs, columns=["p", "q", "r", "s"], index=FitIndxs)
    return Fitness, FitIndxs, pqrsData, maxf_ind

def calcSelection(Fitness):
    """Подсчет итоговой выборки"""
    classification_Table = np.zeros((len(Fitness),len(Fitness)))

    for i in range(len(Fitness)):
        for j in range(i,len(Fitness)):
            if i == j:
                classification_Table[i,j] = 0
                continue  
            if (Fitness[i][0]>Fitness[j][0]):
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
    return selection

def normSelection(array):
    """Нормирование по макс. значению в столбце начиная со 2-го столбца"""
    array = np.array(array)
    maxCols = []
    for i in range(1, len(array[0])):
        max = np.max(np.abs(array[:,i]))
        array[:,i]/=max
        maxCols.append(max)
    return array, maxCols


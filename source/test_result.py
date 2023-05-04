import source.param as param
import source.machine_learning as ml
import pandas as pd
from numpy import sqrt
from sklearn.metrics.pairwise import cosine_similarity

a_j = param.alpha_j
g_j = param.gamma_j
b_j = param.beta_j
d_j = param.delta_j
a_a = param.alpha_a
g_a = param.gamma_a
b_a = param.beta_a
d_a = param.delta_a

def getDerivatives(p, q, r, s):
    """Считаем частные производные в конкретной точке для разложения в ряд Тейлора"""
    hp = -1 + (4*r + 2*(p + q - s))/(2*sqrt(4*p*r + (p + q - s)**2))
    hq = -1 + (p + q - s)/sqrt(4*p*r + (p + q - s)**2)
    hr = (2*p)/sqrt(4*p*r + (p + q - s)**2)
    hs = -1 - (p + q - s)/sqrt(4*p*r + (p + q - s)**2)
    hpp = -(4*r*(q + r - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hpq = (2*r*(p - q + s))/(4*p*r + (p + q - s)**2)**(3/2)
    hpr = (2*((q - s)**2 + p*(q + 2*r - s)))/(4*p*r + (p + q - s)**2)**(3/2)
    hps = -(2*r*(p - q + s))/(4*p*r + (p + q - s)**2)**(3/2)
    hqq = (4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)
    hqr = -(2*p*(p + q - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hqs = -(4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)
    hrr = -(4*p**2)/(4*p*r + (p + q - s)**2)**(3/2)
    hrs = (2*p*(p + q - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hss = (4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)

    return hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss

def findBestPoint(MData, machineCoefs):
    """Ищем лучшую точку для разложения по Тейлору, сравнивая косинусы между векторами вычисленных коэффициентов и полученным вектором машинных"""
    M1 = MData['M1'].tolist()
    M2 = MData['M2'].tolist()      
    M3 = MData['M3'].tolist()   
    M4 = MData['M4'].tolist()   
    M5 = MData['M5'].tolist()
    M6 = MData['M6'].tolist()
    M7 = MData['M7'].tolist()
    M8 = MData['M8'].tolist()

    lamCol = []
    for i in range(1,9):
        lamCol.append('lam'+str(i))
    for i in range(1,9):
        for j in range(i,9):
            lamCol.append('lam'+str(i)+str(j))

    coefsData = pd.DataFrame({'Coefs': []})
    coefsData.loc[len(coefsData)] = [machineCoefs]
    for i in range(MData.shape[0]):
        p = a_j*M1[i] + b_j*M3[i] + d_j*M4[i]
        r = a_a*M5[i] + b_a*M7[i] + d_a*M8[i]
        q = -g_j*M2[i]
        s = -g_a*M6[i]

        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p, q, r, s)

        # Считаем коэффициенты разложения в данной точке (по строкам: при M1-M8, M11-M18, M22-M28, M33-M38, ..., M88)
        calcCoefs = [hp*a_j, hq*(-g_j), hp*b_j, hp*d_j, hr*a_a, hs*(-g_a), hr*b_a, hr*d_a,
        hpp*a_j**2, hpq*a_j*(-g_j), hpp*2*a_j*b_j, hpp*2*a_j*d_j, hpr*a_j*a_a, hps*a_j*(-g_a), hpr*a_j*b_a, hpr*a_j*d_a,
        hqq*(-g_j)**2, hpq*b_j*(-g_j), hpq*d_j*(-g_j), hqr*(-g_j)*a_a, hqs*(-g_j)*(-g_a), hqr*(-g_j)*b_a, hqr*(-g_j)*d_a,
        hpp*b_j**2, hpp*2*b_j*d_j, hpr*b_j*a_a, hps*b_j*(-g_a), hpr*b_j*b_a, hpr*b_j*d_a,
        hpp*d_j**2, hpr*d_j*a_a, hps*d_j*(-g_a), hpr*d_j*b_a, hpr*d_j*d_a,
        hrr*a_a**2, hrs*a_a*(-g_a), hrr*2*a_a*b_a, hrr*2*a_a*d_a,
        hss*(-g_a)**2, hrs*b_a*(-g_a), hrs*d_a*(-g_a),
        hrr*b_a**2, hrr*2*b_a*d_a,
        hrr*d_a**2]

        coefsData.loc[len(coefsData)] = [calcCoefs]
    
    #coefsData.to_csv("inOut/coefs_data.csv", index=False)
    splitCoefsData = pd.DataFrame(coefsData['Coefs'].tolist(), columns=lamCol)
    splitCoefsData.to_csv("inOut/split_coefs_data.csv", index=True)

    cosines = cosine_similarity(splitCoefsData)[0]
    print(cosines)

    bestCosId = 1
    for i in range(2, MData.shape[0]):
        if (cosines[i] > cosines[bestCosId]):
            bestCosId = i
    return bestCosId

def calcTeylor(MData, bstPntId):
    """Проверка правильно ли считаются производные (при не совсем ужасных косинусах считается +-верно)"""
    trueFit = MData['fit'].tolist()
    M1 = MData['M1'].tolist()
    M2 = MData['M2'].tolist()
    M3 = MData['M3'].tolist()
    M4 = MData['M4'].tolist()
    M5 = MData['M5'].tolist()
    M6 = MData['M6'].tolist()
    M7 = MData['M7'].tolist()
    M8 = MData['M8'].tolist()

    p0 = a_j*M1[bstPntId] + b_j*M3[bstPntId] + d_j*M4[bstPntId]
    r0 = a_a*M5[bstPntId] + b_a*M7[bstPntId] + d_a*M8[bstPntId]
    q0 = -g_j*M2[bstPntId]
    s0 = -g_a*M6[bstPntId]
    hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0)
    fit0 = -s0-p0-q0+(sqrt((4*r0*p0+(p0+q0-s0)**2)))

    taylorFit = []
    for i in range(MData.shape[0]):
        p = a_j*M1[i] + b_j*M3[i] + d_j*M4[i]
        r = a_a*M5[i] + b_a*M7[i] + d_a*M8[i]
        q = -g_j*M2[i]
        s = -g_a*M6[i]

        taylorFit.append(fit0 + hp*(p-p0) + hq*(q-q0) + hr*(r-r0) + hs*(s-s0) + 1/2*(hpp*(p-p0)**2 + hqq*(q-q0)**2 + hrr*(r-r0)**2 + hss*(s-s0)**2 + hpq*(p-p0)*(q-q0) + hpr*(p-p0)*(r-r0) + hps*(p-p0)*(s-s0) + hqr*(q-q0)*(r-r0) + hqs*(q-q0)*(s-s0) + hrs*(r-r0)*(s-s0)))
        print(taylorFit[i], "vs", trueFit[i])

def compareCoefs(bstPntId):
    """Сравниваем векторы"""
    coefsData = pd.read_csv("inOut/split_coefs_data.csv")
    machineLam = coefsData.loc[0].to_list()
    calcLam = coefsData.loc[bstPntId].to_list()
    for i in range(44):
        print(calcLam[i], "vs", machineLam[i])


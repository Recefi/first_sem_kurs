import source.param as param
import pandas as pd
import numpy as np
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
    """Считаем частные производные в конкретной точке (p,q,r,s)"""
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

def getCoefData_1(pqrsData, machineCoefs):
    """Считаем коэф-ты для всех точек (p,q,r,s) с учетом минуса при q,s"""
    lamCol = []
    for i in range(1,9):
        lamCol.append('lam'+str(i))
    for i in range(1,9):
        for j in range(i,9):
            lamCol.append('lam'+str(i)+str(j))
    
    coefs = []
    coefs.append(machineCoefs)
    for i in pqrsData.index:
        p, q, r, s = pqrsData.loc[i]
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

        coefs.append(calcCoefs)
    
    indexes = [-1]
    indexes.extend(pqrsData.index)
    coefData = pd.DataFrame(coefs, columns=lamCol, index=indexes)
    coefData.to_csv("csv/coef_data.csv", index=True)
    return coefData

def getCoefData_2(pqrsData, machineCoefs):
    """Считаем коэф-ты для всех точек (p,q,r,s) без учета минуса при q,s"""
    lamCol = []
    for i in range(1,9):
        lamCol.append('lam'+str(i))
    for i in range(1,9):
        for j in range(i,9):
            lamCol.append('lam'+str(i)+str(j))
    
    coefs = []
    coefs.append(machineCoefs)
    for i in pqrsData.index:
        p, q, r, s = pqrsData.loc[i]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p, q, r, s)

        # Считаем коэффициенты разложения в данной точке (по строкам: при M1-M8, M11-M18, M22-M28, M33-M38, ..., M88)
        calcCoefs = [hp*a_j, hq*g_j, hp*b_j, hp*d_j, hr*a_a, hs*g_a, hr*b_a, hr*d_a,
        hpp*a_j**2, hpq*a_j*g_j, hpp*2*a_j*b_j, hpp*2*a_j*d_j, hpr*a_j*a_a, hps*a_j*g_a, hpr*a_j*b_a, hpr*a_j*d_a,
        hqq*g_j**2, hpq*b_j*g_j, hpq*d_j*g_j, hqr*g_j*a_a, hqs*g_j*g_a, hqr*g_j*b_a, hqr*g_j*d_a,
        hpp*b_j**2, hpp*2*b_j*d_j, hpr*b_j*a_a, hps*b_j*g_a, hpr*b_j*b_a, hpr*b_j*d_a,
        hpp*d_j**2, hpr*d_j*a_a, hps*d_j*g_a, hpr*d_j*b_a, hpr*d_j*d_a,
        hrr*a_a**2, hrs*a_a*g_a, hrr*2*a_a*b_a, hrr*2*a_a*d_a,
        hss*g_a**2, hrs*b_a*g_a, hrs*d_a*g_a,
        hrr*b_a**2, hrr*2*b_a*d_a,
        hrr*d_a**2]

        coefs.append(calcCoefs)
    
    indexes = [-1]
    indexes.extend(pqrsData.index)
    coefData = pd.DataFrame(coefs, columns=lamCol, index=indexes)
    coefData.to_csv("csv/coef_data.csv", index=True)
    return coefData

def findNearPoint(coefData):
    """
    Ищем точку (p,q,r,s), коэф-ты в которой ближе всего к машинным, 
        для этого сравниваем косинусы между векторами вычисленных и вектором машинных коэффициентов
    """
    cosines = cosine_similarity(coefData)[0]
    print("\nкосинусы:\n", cosines)

    bestCosId = 1
    for i in range(2, coefData.shape[0]):
        if (cosines[i] > cosines[bestCosId]):
            bestCosId = i

    nearPntId = coefData.index[bestCosId]
    print("nearPntId =", nearPntId, ": ", cosines[bestCosId])
    return nearPntId

def checkCalcCoef(fitData, pqrsData, maxFitPntId, nearPntId):
    """
    Проверка правильно ли считаются производные
        или же насколько грубо полноценное разложение по Тейлору до 2 порядка аппроксимирует функцию фитнеса
    """
    def fullTaylor(pntId):
        p0, q0, r0, s0 = pqrsData.loc[pntId]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0)

        fit0 = -s0-p0-q0+(sqrt((4*r0*p0+(p0+q0-s0)**2)))
        trueFit0 = fitData.loc[pntId, 'fit']
        if (fit0 != trueFit0):
            print("PQRS_INDEX_ERROR")

        taylorFit = []
        for i in pqrsData.index:
            p, q, r, s = pqrsData.loc[i]
            taylorFit.append(fit0 + hp*(p-p0) + hq*(q-q0) + hr*(r-r0) + hs*(s-s0) + 1/2*(hpp*(p-p0)**2 + hqq*(q-q0)**2 + hrr*(r-r0)**2 + hss*(s-s0)**2 + hpq*(p-p0)*(q-q0) + hpr*(p-p0)*(r-r0) + hps*(p-p0)*(s-s0) + hqr*(q-q0)*(r-r0) + hqs*(q-q0)*(s-s0) + hrs*(r-r0)*(s-s0)))
        return taylorFit
    
    def shortTaylor(pntId):
        p0, q0, r0, s0 = pqrsData.loc[pntId]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0)

        fit0 = -s0-p0-q0+(sqrt((4*r0*p0+(p0+q0-s0)**2)))
        trueFit0 = fitData.loc[pntId, 'fit']
        if (fit0 != trueFit0):
            print("PQRS_INDEX_ERROR")

        taylorFit = []
        for i in pqrsData.index:
            p, q, r, s = pqrsData.loc[i]
            taylorFit.append(hp*p + hq*q + hr*r + hs*s + hpp*p**2 + hqq*q**2 + hrr*r**2 + hss*s**2 + hpq*p*q + hpr*p*r + hps*p*s + hqr*q*r + hqs*q*s + hrs*r*s)
        return taylorFit
    
    trueFit = fitData['fit']
    fT_maxFitPnt = fullTaylor(maxFitPntId)
    shT_maxFitPnt = shortTaylor(maxFitPntId)
    fT_nearPnt = fullTaylor(nearPntId)
    shT_nearPnt = shortTaylor(nearPntId)

    checkCoefData = pd.DataFrame({'trueFit': trueFit, 'fT_maxFitPnt': fT_maxFitPnt, 'shT_maxFitPnt': shT_maxFitPnt, 
                                  'fT_nearPnt': fT_nearPnt, 'shT_nearPnt': shT_nearPnt}, index=trueFit.index)
    return checkCoefData

def compareCoefs(coefData, nearPntId):
    """Сравниваем коэффициенты"""
    machCoef = coefData.loc[-1]
    calcCoef = coefData.loc[nearPntId]
    for i in range(44):
        print(coefData.columns[i], machCoef[i], "vs", calcCoef[i])

def compareNormCoefs(coefData, nearPntId):
    """Сравниваем нормированные коэффициенты"""
    machCoef = coefData.loc[-1].copy()
    calcCoef = coefData.loc[nearPntId].copy()
    #print(calcCoef.index)
    
    machCoef/=(np.max(np.abs(machCoef)))
    calcCoef/=(np.max(np.abs(calcCoef)))

    for i in range(44):
        print(coefData.columns[i], machCoef[i], "vs", calcCoef[i])


import numpy as np
import pandas as pd
import source.gen_selection as gs
import source.graphical_interface as gui
import source.input_output as inOut


# Либо
Aj, Bj, Aa, Ba = gs.genStrats(5000)
stratData = inOut.collectStratData(Aj, Bj, Aa, Ba)
inOut.writeData(stratData, "strat_data_2")
# # Либо
# stratData = inOut.readData("strat_data_2")
# Aj, Bj, Aa, Ba = inOut.parseStratData(stratData)


Fitness, FitIndxs, pqrsData, maxFitId = gs.calcFitness(stratData)
trueFits = np.transpose(Fitness)[0]

coefData = inOut.readData("coef_data")
coefs = coefData.loc[-1].tolist()
restoredFits = []
for i in range(len(FitIndxs)):
    Mparam = Fitness[i][1:45]
    fit = 0
    for j in range(44):
        fit += coefs[j]*Mparam[j]
    restoredFits.append(fit)

print(len(trueFits))

checkCoefData = pd.DataFrame({'trueFit': trueFits, 'restoredFit': restoredFits}, index=FitIndxs)
stratFitData = inOut.collectStratFitData(stratData, checkCoefData)
inOut.writeData(stratFitData, "strat_fit_data_2")

maxTrueFitId = checkCoefData[['trueFit']].idxmax(axis='index')[0]
maxRestrFitId = checkCoefData[['restoredFit']].idxmax(axis='index')[0]

gui.show_comparison_sinss(stratData, maxTrueFitId, maxRestrFitId)


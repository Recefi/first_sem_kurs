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
fitData = inOut.collectFitData(Fitness, FitIndxs)

coefData = inOut.readData("coef_data")
coefs = coefData.loc[-1].tolist()
restoredFits = []
for i in fitData.index:
    Mparam = fitData.loc[i, 'M1':'M8M8']
    fit = 0
    for j in range(44):
        fit += coefs[j]*Mparam[j]
    restoredFits.append(fit)

trueFits = fitData['fit']
checkCoefData = pd.DataFrame({'trueFit': trueFits, 'restoredFit': restoredFits}, index=FitIndxs)

maxTrueFitId = checkCoefData[['trueFit']].idxmax(axis='index')[0]
maxRestrFitId = checkCoefData[['restoredFit']].idxmax(axis='index')[0]

gui.show_comparison_sinss(stratData, maxTrueFitId, maxRestrFitId)


stratFitData = inOut.collectStratFitData(stratData, fitData)
inOut.writeData(stratFitData, "strat_fit_data")
fitDataByAbsVals = inOut.collectFitDataByAbsVals(fitData)
inOut.writeData(fitDataByAbsVals, "fit_data_byAbsVals")

stratBothFitData = inOut.collectStratBothFitData(stratData, checkCoefData)
inOut.writeData(stratBothFitData, "strat_both_fit_data")
fitDataByAbsVals = inOut.collectBothFitDataByAbsVals(checkCoefData)
inOut.writeData(fitDataByAbsVals, "both_fit_data_byAbsVals")

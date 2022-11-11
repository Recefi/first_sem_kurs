import source.gen_selection as gs
import source.graphical_interface as gui
import numpy as np


row_table = 20 #Количество строк в таблицах
gui.get_table('Macroparams',['M1','M2','M3','M4','M5','M6','M7','M8'],[[i] for i in range(row_table)],np.transpose(gs.Macroparameters)[:row_table])
gui.get_table('Norm macroparams',['M1','M2','M3','M4','M5','M6','M7','M8'],[[i] for i in range(row_table)],np.transpose(gs.norm_Macroparameters)[:row_table])

for i in range(len(gs.norm_Macroparameters)):
    gui.get_gistogram(gs.norm_Macroparameters[i],"M"+str(i+1))

gui.get_correllation(gs.norm_Macroparameters,["M1","M2","M3","M4","M5","M6","M7","M8"])
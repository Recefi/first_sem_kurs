import source.gen_selection as gs
import source.graphical_interface as gui
import numpy as np
import source.machine_learning as ml
import source.test_result as tr



#row_table = 20 #Количество строк в таблицах
#gui.get_table('A',['A'],[[i] for i in range(row_table)],np.transpose(gs.A_jun)[:row_table])
# gui.get_table('Norm macroparams',['M1','M2','M3','M4','M5','M6','M7','M8'],[[i] for i in range(row_table)],np.transpose(gs.norm_Macroparameters)[:row_table])

for i in range(8):
    gui.get_gistogram(gs.selection[i],"M"+str(i+1))
#print(gs.A_jun[gs.maxf_ind])
#print(gs.B_jun[gs.maxf_ind])
gui.get_sinss()

gui.get_correllation(np.transpose(gs.Fitness)[1:9:1],["M1","M2","M3","M4","M5","M6","M7","M8"])


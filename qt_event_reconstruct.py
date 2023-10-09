from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy
#from qt_events_compression import *

def qt_event_reconstruct(chip_host_events, s_a_final, Nbins_T, tree, sq, event_3D_last):

    event_3D_rec = np.zeros(shape=(512, 512, Nbins_T))
    Nl = len(sq)
    N_tau = np.size(tree, 0)

    for k in range(1, N_tau+1):
        b = sq[Nl - int(tree[k-1, 0])][:, int(tree[k-1, 1])-1]

        #print('chip_host_events[k-1]', chip_host_events[k-1])

        if (s_a_final == 0):
            # this is acquire events for both skip-acquire of intensity 
            if len(np.shape(chip_host_events[k-1])) > 1:
                chi_host_act_ext = chip_host_events[k-1]
                chi_host_act = chi_host_act_ext[:, :2] + np.array([int(b[0]) - 1, int(b[1]) - 1]) 
                for ii in range(np.shape(chi_host_act)[0]):
                    event_3D_rec[int(chi_host_act[ii, 0]), int(chi_host_act[ii, 1]), int(chi_host_act_ext[ii, 2])] = int(chi_host_act_ext[ii, 3])
    
        elif (s_a_final == 1):
            if tree[k-1, 2] == 2:
                if len(np.shape(chip_host_events[k-1])) > 1:
                    chi_host_act_ext = chip_host_events[k-1]
                    chi_host_act = chi_host_act_ext[:, :2] + np.array([int(b[0]) - 1, int(b[1]) - 1])
                    for ii in range(np.shape(chi_host_act)[0]):
                        event_3D_rec[int(chi_host_act[ii, 0]), int(chi_host_act[ii, 1]), int(chi_host_act_ext[ii, 2])] = int(chi_host_act_ext[ii, 3])
            else:
                event_3D_rec[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :] = np.copy(event_3D_last[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :])        


    return event_3D_rec 

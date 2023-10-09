from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy
from geometric_median import *
from scipy.spatial import distance
from rle import *
import cv2
import hdf5storage
from skimage import measure

def qt_events(tree, sq, event_file_c, r_4, s_a, comp2, event_3D_last, r_1, r_2, k123, write_huff_tables):
    
    
    print('r_4, r_1, r_2, s_a', r_4, r_1, r_2, s_a)
    ppp = 0
    
    # load the Huffman tables for checking compression
    huff_2_x = np.loadtxt('final_huff_tables_Aug31/huff_2_x.txt', delimiter = ',', dtype='str'); huff_2_x1 = huff_2_x[:, 0].astype(int)
    huff_2_y = np.loadtxt('final_huff_tables_Aug31/huff_2_y.txt', delimiter = ',', dtype='str'); huff_2_y1 = huff_2_y[:, 0].astype(int)
    huff_4_x = np.loadtxt('final_huff_tables_Aug31/huff_4_x.txt', delimiter = ',', dtype='str'); huff_4_x1 = huff_4_x[:, 0].astype(int)
    huff_4_y = np.loadtxt('final_huff_tables_Aug31/huff_4_y.txt', delimiter = ',', dtype='str'); huff_4_y1 = huff_4_y[:, 0].astype(int)
    huff_8_x = np.loadtxt('final_huff_tables_Aug31/huff_8_x.txt', delimiter = ',', dtype='str'); huff_8_x1 = huff_8_x[:, 0].astype(int)
    huff_8_y = np.loadtxt('final_huff_tables_Aug31/huff_8_y.txt', delimiter = ',', dtype='str'); huff_8_y1 = huff_8_y[:, 0].astype(int)
    huff_16_x = np.loadtxt('final_huff_tables_Aug31/huff_16_x.txt', delimiter = ',', dtype='str'); huff_16_x1 = huff_16_x[:,0].astype(int)
    huff_16_y = np.loadtxt('final_huff_tables_Aug31/huff_16_y.txt', delimiter = ',', dtype='str'); huff_16_y1 = huff_16_y[:,0].astype(int)
    huff_32_x = np.loadtxt('final_huff_tables_Aug31/huff_32_x.txt', delimiter = ',', dtype='str'); huff_32_x1 = huff_32_x[:, 0].astype(int)
    huff_32_y = np.loadtxt('final_huff_tables_Aug31/huff_32_y.txt', delimiter = ',', dtype = 'str'); huff_32_y1 = huff_32_y[:, 0].astype(int)
    
    '''
    huff_64_x = np.loadtxt('final_huff_tables_Aug31/huff_64_x.txt', delimiter = ',', dtype = 'str'); huff_64_x1 = huff_64_x[:, 0].astype(int)
    huff_64_y = np.loadtxt('final_huff_tables_Aug31/huff_64_y.txt', delimiter = ',', dtype = 'str'); huff_64_y1 = huff_64_y[:, 0].astype(int)
    '''

    huff_1_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_1_p_pos.txt', delimiter = ',', dtype='str'); huff_1_p_pos1 = huff_1_p_pos[:, 0].astype(int)
    huff_1_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_1_p_neg.txt', delimiter = ',', dtype='str'); huff_1_p_neg1 = huff_1_p_neg[:, 0].astype(int)
    huff_2_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_2_p_pos.txt', delimiter = ',', dtype='str'); huff_2_p_pos1 = huff_2_p_pos[:, 0].astype(int)
    huff_2_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_2_p_neg.txt', delimiter = ',', dtype='str'); huff_2_p_neg1 = huff_2_p_neg[:, 0].astype(int)
    huff_4_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_4_p_pos.txt', delimiter = ',', dtype='str'); huff_4_p_pos1 = huff_4_p_pos[:, 0].astype(int)
    huff_4_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_4_p_neg.txt', delimiter = ',', dtype='str'); huff_4_p_neg1 = huff_4_p_neg[:, 0].astype(int)
    huff_8_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_8_p_pos.txt', delimiter = ',', dtype='str'); huff_8_p_pos1 = huff_8_p_pos[:, 0].astype(int)
    huff_8_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_8_p_neg.txt', delimiter = ',', dtype='str'); huff_8_p_neg1 = huff_8_p_neg[:, 0].astype(int)
    huff_16_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_16_p_pos.txt', delimiter = ',', dtype='str'); huff_16_p_pos1 = huff_16_p_pos[:, 0].astype(int)
    huff_16_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_16_p_neg.txt', delimiter = ',', dtype='str'); huff_16_p_neg1 = huff_16_p_neg[:, 0].astype(int)
    huff_32_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_32_p_pos.txt', delimiter = ',', dtype='str'); huff_32_p_pos1 = huff_32_p_pos[:, 0].astype(int)
    huff_32_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_32_p_neg.txt', delimiter = ',', dtype='str'); huff_32_p_neg1 = huff_32_p_neg[:, 0].astype(int)
    
    '''
    huff_64_p_pos = np.loadtxt('final_huff_tables_Aug31/rle_64_p_pos.txt', delimiter = ',', dtype='str'); huff_64_p_pos1 = huff_64_p_pos[:, 0].astype(int)
    huff_64_p_neg = np.loadtxt('final_huff_tables_Aug31/rle_64_p_neg.txt', delimiter = ',', dtype='str'); huff_64_p_neg1 = huff_64_p_neg[:, 0].astype(int)
    '''

    T_50 = 1/50
    '''create 3D volume of events'''
    events_load = hdf5storage.loadmat(event_file_c)
    kkey = events_load.keys()[0]
    events_extr = np.asarray(events_load[kkey])
    events_extr_act = events_extr[0]
    events_extr_x = events_extr_act[0].squeeze(); events_extr_y = events_extr_act[1].squeeze()
    events_extr_p = events_extr_act[2].squeeze(); events_extr_t = events_extr_act[3].squeeze()
    events = np.column_stack((events_extr_y, events_extr_x, events_extr_t, events_extr_p))
    TOTAL_EVENTS_ORG = np.shape(events)[0] 

    # events represented as x,y,t,p
    T_min = np.min(events[:, 2])
    #print('ev_t max and min', np.max(events[:, 2]), np.min(events[:, 2]))
    events[:, 2] = events[:, 2] - T_min
    #print('events_max and events_min', np.max(events[:,2]), np.min(events[:,2]))
    print('events shape', np.shape(events))
    events_org = np.copy(events)

    # Number of time bins
    Nbins_T = 16
    T_disc = np.linspace(0, T_50, Nbins_T+1)
    events[:,2] = np.digitize(events[:,2], T_disc) # quantize in time straight away
 
    event_3D_pos = np.zeros(shape=(256, 256, Nbins_T)); event_3D_neg = np.zeros(shape=(256, 256, Nbins_T))
    event_3D_new = np.zeros(shape=(256, 256, Nbins_T))

    for ii in range(np.shape(events)[0]):
        if events[ii, 3] > 0.8:
            event_3D_pos[np.int(events[ii, 0])+38, np.int(events[ii, 1])+33, np.int(events[ii, 2]) - 1] += np.int(events[ii, 3])
        else:
            event_3D_neg[np.int(events[ii, 0])+38, np.int(events[ii, 1])+33, np.int(events[ii, 2]) - 1] += np.int(events[ii, 3])

    print('org_pos_events', np.count_nonzero(event_3D_pos))
    print('org_neg_events', np.count_nonzero(event_3D_neg))
    
    td_image_org = np.zeros((256, 256), dtype=np.uint8)
    td_image_org = np.sum(np.abs(event_3D_pos), axis=2) + np.sum(np.abs(event_3D_neg), axis=2)
    
    '''
    td_image_org1 = np.clip(td_image_org, 0, 1)
    td_image_org2 = np.where(td_image_org1 == 1, 255, td_image_org1)
    td_image_org2 = td_image_org2.astype(dtype = np.uint8)

    td_image_org2_save = cv2.cvtColor(td_image_org2, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('event_org_region.jpg', td_image_org2_save)
    print('count original events', np.count_nonzero(event_3D))
    '''

    '''both are same'''
    #TOTAL_EVENTS_ORG = np.count_nonzero(event_3D_pos) + np.count_nonzero(event_3D_neg)

    BIT_STD1 = TOTAL_EVENTS_ORG * 64
    BIT_STD2 = TOTAL_EVENTS_ORG * 96
    BIT_ORG = TOTAL_EVENTS_ORG * 22 #each event is 22 bits
    print('event length', TOTAL_EVENTS_ORG)
    
    TOT_COMPR_BITS = 0

    Nl = len(sq)
    N_tau = np.size(tree,0)
    #print('N_tau q_events', N_tau)
    #print('tree_few', tree[:10, :])
    event_e_rec = np.zeros((N_tau,1))

    # calculate values of all possible leaves from acquired image
    a2 = -1*np.ones((N_tau,1))
    tree_rec = np.hstack([tree[0:N_tau,:], a2])

    event_bits = 0
    total_events = 0
    
    #ch_h_l = np.count_nonzero(tree[:,2] == 2)
    ch_h_l = np.shape(tree)[0]
    chip_host_events = [[0] for x in range(ch_h_l)]

    chip_host_events_1x1_pos = []; chip_host_events_1x1_neg = []
    rle_64_all_p_pos = []; rle_64_all_p_neg = []; rle_32_all_p_pos = []; rle_32_all_p_neg = []; rle_16_all_p_pos = []; rle_16_all_p_neg = []; rle_8_all_p_pos = []; rle_8_all_p_neg = [] 
    rle_4_all_p_pos = []; rle_4_all_p_neg = []; rle_2_all_p_pos = []; rle_2_all_p_neg = []
    '''huff_64_xy = []; huff_32_xy = []; huff_16_xy = []; huff_8_xy = []; huff_4_xy = []; huff_2_xy = []'''

    block_size_4 = 4 

    neighbors_4x4 = []
    for xx in range(np.int(np.round(-r_4)), np.int(np.round(r_4))+1):
        for yy in range(np.int(np.round(-r_4)), np.int(np.round(r_4))+1):
            if (xx**2) + (yy**2) <= (r_4**2):
                if (xx == 0) and (yy == 0):
                    continue
                neighbors_4x4.append([xx, yy])
    
    r_8 = 2*r_4
    block_size_8 = 8

    neighbors_8x8 = []
    for xx in range(np.int(np.round(-r_8)), np.int(np.round(r_8))+1):
        for yy in range(np.int(np.round(-r_8)), np.int(np.round(r_8))+1):
            if (xx**2) + (yy**2) <= (r_8**2):
                if (xx == 0) and (yy == 0):
                    continue
                neighbors_8x8.append([xx, yy])
 
    r_16 = 3*r_4
    block_size_16 = 16

    neighbors_16x16 = []
    for xx in range(np.int(np.round(-r_16)), np.int(np.round(r_16))+1):
        for yy in range(np.int(np.round(-r_16)), np.int(np.round(r_16))+1):
            if (xx**2) + (yy**2) <= (r_16**2):
                if (xx**2) + (yy**2) <= (r_16**2):
                    if (xx == 0) and (yy == 0):
                        continue
                    neighbors_16x16.append([xx, yy])
    
    r_32 = 4*r_4
    block_size_32 = 32

    neighbors_32x32 = []
    for xx in range(np.int(np.round(-r_32)), np.int(np.round(r_32))+1):
        for yy in range(np.int(np.round(-r_32)), np.int(np.round(r_32))+1):
            if (xx**2) + (yy**2) <= (r_32**2):
                if (xx==0) and (yy==0):
                    continue
                neighbors_32x32.append([xx, yy])

    r_64 = 5*r_4
    block_size_64 = 64

    neighbors_64x64 = []
    for xx in range(np.int(np.round(-r_64)), np.int(np.round(r_64))+1):
        for yy in range(np.int(np.round(-r_64)), np.int(np.round(r_64))+1):
            if (xx**2) + (yy**2) <= (r_64**2):
                if (xx==0) and (yy==0):
                    continue
                neighbors_64x64.append([xx, yy])
    
    
    if (len(neighbors_8x8) == 0):
        ppp = ppp + 1

    if (len(neighbors_4x4) == 0):
        ppp = ppp + 1

    for k in range(1,N_tau+1):
        b = sq[Nl - int(tree[k-1,0])][:,int(tree[k-1,1])-1]
        '''this gives the 2d co-ordinates of the QT in intensity'''
        '''the events are extracted from the same co-ordinates'''
        #print('b', b)
        #print('tree[k-1,:]', tree[k-1, :])
        
        '''
        if (s_a == 1) and (tree[k-1,2] == 1):
            this is for copying the skip modes only
            event_3D_new[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :] = np.copy(event_3D_last[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :]) 
            chip_host_events[k-1] = [0]
        '''

        #print('b[2], b[3]', int(b[2]) - int(b[0]) + 1, int(b[3]) - int(b[1]) + 1)

        if ((s_a==0) and ((tree[k-1,2] == 2) or (tree[k-1, 2] == 1))) or ((s_a==1) and (tree[k-1,2] == 2)):
            '''this is for both skip and acquire modes (everything) '''


            if ((int(b[2]) - int(b[0]) + 1) == 64) and ((int(b[3]) - int(b[1]) + 1) == 64):
                    event_extr_64_pos = np.copy(event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                    event_extr_64_ccc_pos = np.count_nonzero(event_extr_64_pos)
  
                    if event_extr_64_ccc_pos > 0:
                        for zz in range(Nbins_T):
                            event_extr_64_zz_pos = event_extr_64_pos[:, :, zz]
                            if np.count_nonzero(event_extr_64_zz_pos) > 0:
                                 nz_ind_zz_64_pos = np.transpose(np.nonzero(event_extr_64_zz_pos))
                                 event_extr_median_64_pos = geometric_median(nz_ind_zz_64_pos, 'auto')
                                 D_64_pos = distance.cdist(np.array([event_extr_median_64_pos]), nz_ind_zz_64_pos, 'euclidean')
                                 D_64_min_ind_pos = np.argmin(D_64_pos)
                                 nz_ind_zz_median_r_64_pos = nz_ind_zz_64_pos[D_64_min_ind_pos, :]
                                 nz_ind_del_64_pos = np.delete(nz_ind_zz_64_pos, D_64_min_ind_pos, axis=0)
   
                                 while (len(nz_ind_del_64_pos) > 0) and (len(neighbors_64x64) > 0):
                                     nz_median_64_neighbors_pos = nz_ind_zz_median_r_64_pos + neighbors_64x64
   
                                     for kk in range(np.shape(neighbors_64x64)[0]):
                                         if (((nz_median_64_neighbors_pos[kk, 0] >= 0) and (nz_median_64_neighbors_pos[kk, 0] < block_size_64)) and\
                                             ((nz_median_64_neighbors_pos[kk, 1] >= 0) and (nz_median_64_neighbors_pos[kk, 1] < block_size_64))):
   
                                             event_extr_64_zz_pos[int(nz_median_64_neighbors_pos[kk,0]), int(nz_median_64_neighbors_pos[kk,1])] = 0
                                             event_extr_64_pos[:, :, zz] = event_extr_64_zz_pos
   
                                             cc_64_pos = np.where(np.all(nz_median_64_neighbors_pos[kk,:] == nz_ind_del_64_pos, axis=1))
                                             nz_ind_del_64_pos = np.delete(nz_ind_del_64_pos, cc_64_pos, axis=0)
   
                                     D1_64_pos = distance.cdist(np.array([nz_ind_zz_median_r_64_pos]), nz_ind_del_64_pos, 'euclidean')
                                     D1_sort_64_pos = np.argsort(D1_64_pos)
   
   
                                     if np.shape(D1_sort_64_pos)[1] > 0:
                                         del_ind_64_pos = D1_sort_64_pos[0, 0]
                                         nz_ind_zz_median_r_64_pos = nz_ind_del_64_pos[del_ind_64_pos, :]
                                         nz_ind_del_64_pos = np.delete(nz_ind_del_64_pos, del_ind_64_pos, axis=0)
                                     else:
                                         break

                        event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_64_pos
                        
                        '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_64'''
                        
                        event_extr_64_nz_ind_pos = np.nonzero(event_extr_64_pos)
                        event_extr_64_nz_c_pos = np.column_stack((event_extr_64_nz_ind_pos[0], event_extr_64_nz_ind_pos[1], event_extr_64_nz_ind_pos[2], event_extr_64_pos[event_extr_64_nz_ind_pos]))
                        event_extr_64_pos_yx = event_extr_64_nz_c_pos[:, :2]; event_extr_64_pos_p = event_extr_64_nz_c_pos[:, -1]                       
                        rle_64_p_pos, rle_64_p_bits_pos, rle_64_p_len_bits_pos = rle(event_extr_64_pos_p, binary = False)

                        for zzz in range(np.shape(event_extr_64_pos_yx)[0]):
                            match_ind_y = np.argwhere(int(event_extr_64_pos_yx[zzz, 0]) == huff_64_y1)
                            TOT_COMPR_BITS += len(huff_64_y[match_ind_y, 1])

                            match_ind_x = np.argwhere(int(event_extr_64_pos_yx[zzz, 1]) == huff_64_x1)
                            TOT_COMPR_BITS += len(huff_64_x[match_ind_x, 1])
 
                        for yyy in range(np.shape(rle_64_p_pos)[0]):
                            match_ind_p_pos = np.argwhere(int(rle_64_p_pos[yyy, 1]) == huff_64_p_pos1)
                            TOT_COMPR_BITS += len(huff_64_p_pos[match_ind_p_pos, 1])


            if ((int(b[2]) - int(b[0]) + 1) == 64) and ((int(b[3]) - int(b[1]) + 1) == 64):
                    event_extr_64_neg = np.copy(event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                    event_extr_64_ccc_neg = np.count_nonzero(event_extr_64_neg)

                    if event_extr_64_ccc_neg > 0:
                        for zz in range(Nbins_T):
                            event_extr_64_zz_neg = event_extr_64_neg[:, :, zz]
                            if np.count_nonzero(event_extr_64_zz_neg) > 0:
                                 nz_ind_zz_64_neg = np.transpose(np.nonzero(event_extr_64_zz_neg))
                                 event_extr_median_64_neg = geometric_median(nz_ind_zz_64_neg, 'auto')
                                 D_64_neg = distance.cdist(np.array([event_extr_median_64_neg]), nz_ind_zz_64_neg, 'euclidean')
                                 D_64_min_ind_neg = np.argmin(D_64_neg)
                                 nz_ind_zz_median_r_64_neg = nz_ind_zz_64_neg[D_64_min_ind_neg, :]
                                 nz_ind_del_64_neg = np.delete(nz_ind_zz_64_neg, D_64_min_ind_neg, axis=0)

                                 while (len(nz_ind_del_64_neg) > 0) and (len(neighbors_64x64) > 0):
                                     nz_median_64_neighbors_neg = nz_ind_zz_median_r_64_neg + neighbors_64x64

                                     for kk in range(np.shape(neighbors_64x64)[0]):
                                         if (((nz_median_64_neighbors_neg[kk, 0] >= 0) and (nz_median_64_neighbors_neg[kk, 0] < block_size_64)) and\
                                             ((nz_median_64_neighbors_neg[kk, 1] >= 0) and (nz_median_64_neighbors_neg[kk, 1] < block_size_64))):

                                             event_extr_64_zz_neg[int(nz_median_64_neighbors_neg[kk,0]), int(nz_median_64_neighbors_neg[kk,1])] = 0
                                             event_extr_64_neg[:, :, zz] = event_extr_64_zz_neg

                                             cc_64_neg = np.where(np.all(nz_median_64_neighbors_neg[kk,:] == nz_ind_del_64_neg, axis=1))
                                             nz_ind_del_64_neg = np.delete(nz_ind_del_64_neg, cc_64_neg, axis=0)
 
                                     D1_64_neg = distance.cdist(np.array([nz_ind_zz_median_r_64_neg]), nz_ind_del_64_neg, 'euclidean')
                                     D1_sort_64_neg = np.argsort(D1_64_neg)
 

                                     if np.shape(D1_sort_64_neg)[1] > 0:
                                         del_ind_64_neg = D1_sort_64_neg[0, 0]
                                         nz_ind_zz_median_r_64_neg = nz_ind_del_64_neg[del_ind_64_neg, :]
                                         nz_ind_del_64_neg = np.delete(nz_ind_del_64_neg, del_ind_64_neg, axis=0)
                                     else:
                                         break
 
                        event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_64_neg

                        '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_64'''

                        event_extr_64_nz_ind_neg = np.nonzero(event_extr_64_neg)
                        event_extr_64_nz_c_neg = np.column_stack((event_extr_64_nz_ind_neg[0], event_extr_64_nz_ind_neg[1], event_extr_64_nz_ind_neg[2], event_extr_64_neg[event_extr_64_nz_ind_neg]))
                        
                        event_extr_64_neg_yx = event_extr_64_nz_c_neg[:, :2]; event_extr_64_neg_p = event_extr_64_nz_c_neg[:, -1]
                        rle_64_p_neg, rle_64_p_bits_neg, rle_64_p_len_bits_neg = rle(event_extr_64_neg_p, binary = False)
                         
                        for zzz in range(np.shape(event_extr_64_neg_yx)[0]):
                            match_ind_y = np.argwhere(int(event_extr_64_neg_yx[zzz, 0]) == huff_64_y1)
                            TOT_COMPR_BITS += len(huff_64_y[match_ind_y, 1])
                             
                            match_ind_x = np.argwhere(int(event_extr_64_neg_yx[zzz, 1]) == huff_64_x1)
                            TOT_COMPR_BITS += len(huff_64_x[match_ind_x, 1])
                             
                        for yyy in range(np.shape(rle_64_p_neg)[0]):
                            match_ind_p_neg = np.argwhere(int(rle_64_p_neg[yyy, 1]) == huff_64_p_neg1)
                            TOT_COMPR_BITS += len(huff_64_p_neg[match_ind_p_neg, 1])


            if ((int(b[2]) - int(b[0]) + 1) == 32) and ((int(b[3]) - int(b[1]) + 1) == 32):
                event_extr_32_pos = np.copy(event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_32_ccc_pos = np.count_nonzero(event_extr_32_pos)

                if event_extr_32_ccc_pos > 0:
                    for zz in range(Nbins_T):
                        event_extr_32_zz_pos = event_extr_32_pos[:, :, zz]
                        if np.count_nonzero(event_extr_32_zz_pos) > 0:
                             nz_ind_zz_32_pos = np.transpose(np.nonzero(event_extr_32_zz_pos))
                             event_extr_median_32_pos = geometric_median(nz_ind_zz_32_pos, 'auto')
                             D_32_pos = distance.cdist(np.array([event_extr_median_32_pos]), nz_ind_zz_32_pos, 'euclidean')
                             D_32_min_ind_pos = np.argmin(D_32_pos)
                             nz_ind_zz_median_r_32_pos = nz_ind_zz_32_pos[D_32_min_ind_pos, :]
                             nz_ind_del_32_pos = np.delete(nz_ind_zz_32_pos, D_32_min_ind_pos, axis=0)

                             while (len(nz_ind_del_32_pos) > 0) and (len(neighbors_32x32) > 0):
                                 nz_median_32_neighbors_pos = nz_ind_zz_median_r_32_pos + neighbors_32x32

                                 for kk in range(np.shape(neighbors_32x32)[0]):
                                     if (((nz_median_32_neighbors_pos[kk, 0] >= 0) and (nz_median_32_neighbors_pos[kk, 0] < block_size_32)) and\
                                         ((nz_median_32_neighbors_pos[kk, 1] >= 0) and (nz_median_32_neighbors_pos[kk, 1] < block_size_32))):

                                         event_extr_32_zz_pos[int(nz_median_32_neighbors_pos[kk,0]), int(nz_median_32_neighbors_pos[kk,1])] = 0
                                         event_extr_32_pos[:, :, zz] = event_extr_32_zz_pos

                                         cc_32_pos = np.where(np.all(nz_median_32_neighbors_pos[kk,:] == nz_ind_del_32_pos, axis=1))
                                         nz_ind_del_32_pos = np.delete(nz_ind_del_32_pos, cc_32_pos, axis=0)

                                 D1_32_pos = distance.cdist(np.array([nz_ind_zz_median_r_32_pos]), nz_ind_del_32_pos, 'euclidean')
                                 D1_sort_32_pos = np.argsort(D1_32_pos)


                                 if np.shape(D1_sort_32_pos)[1] > 0:
                                     del_ind_32_pos = D1_sort_32_pos[0, 0]
                                     nz_ind_zz_median_r_32_pos = nz_ind_del_32_pos[del_ind_32_pos, :]
                                     nz_ind_del_32_pos = np.delete(nz_ind_del_32_pos, del_ind_32_pos, axis=0)
                                 else:
                                     break

                    event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_32_pos
                    
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_32'''
                    
                    event_extr_32_nz_ind_pos = np.nonzero(event_extr_32_pos)
                    event_extr_32_nz_c_pos = np.column_stack((event_extr_32_nz_ind_pos[0], event_extr_32_nz_ind_pos[1], event_extr_32_nz_ind_pos[2], event_extr_32_pos[event_extr_32_nz_ind_pos]))
                    
                    event_extr_32_pos_yx = event_extr_32_nz_c_pos[:, :2]; event_extr_32_pos_p = event_extr_32_nz_c_pos[:, -1]
                    rle_32_p_pos, rle_32_p_bits_pos, rle_32_p_len_bits_pos = rle(event_extr_32_pos_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_32_pos_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_32_pos_yx[zzz, 0]) == huff_32_y1)
                        TOT_COMPR_BITS += len(huff_32_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_32_pos_yx[zzz, 1]) == huff_32_x1)
                        TOT_COMPR_BITS += len(huff_32_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_32_p_pos)[0]):
                        match_ind_p_pos = np.argwhere(int(rle_32_p_pos[yyy, 1]) == huff_32_p_pos1)
                        TOT_COMPR_BITS += len(huff_32_p_pos[match_ind_p_pos, 1])

            if ((int(b[2]) - int(b[0]) + 1) == 32) and ((int(b[3]) - int(b[1]) + 1) == 32):
                event_extr_32_neg = np.copy(event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_32_ccc_neg = np.count_nonzero(event_extr_32_neg)

                if event_extr_32_ccc_neg > 0:
                    for zz in range(Nbins_T):
                        event_extr_32_zz_neg = event_extr_32_neg[:, :, zz]
                        if np.count_nonzero(event_extr_32_zz_neg) > 0:
                             nz_ind_zz_32_neg = np.transpose(np.nonzero(event_extr_32_zz_neg))
                             event_extr_median_32_neg = geometric_median(nz_ind_zz_32_neg, 'auto')
                             D_32_neg = distance.cdist(np.array([event_extr_median_32_neg]), nz_ind_zz_32_neg, 'euclidean')
                             D_32_min_ind_neg = np.argmin(D_32_neg)
                             nz_ind_zz_median_r_32_neg = nz_ind_zz_32_neg[D_32_min_ind_neg, :]
                             nz_ind_del_32_neg = np.delete(nz_ind_zz_32_neg, D_32_min_ind_neg, axis=0)

                             while (len(nz_ind_del_32_neg) > 0) and (len(neighbors_32x32) > 0):
                                 nz_median_32_neighbors_neg = nz_ind_zz_median_r_32_neg + neighbors_32x32

                                 for kk in range(np.shape(neighbors_32x32)[0]):
                                     if (((nz_median_32_neighbors_neg[kk, 0] >= 0) and (nz_median_32_neighbors_neg[kk, 0] < block_size_32)) and\
                                         ((nz_median_32_neighbors_neg[kk, 1] >= 0) and (nz_median_32_neighbors_neg[kk, 1] < block_size_32))):

                                         event_extr_32_zz_neg[int(nz_median_32_neighbors_neg[kk,0]), int(nz_median_32_neighbors_neg[kk,1])] = 0
                                         event_extr_32_neg[:, :, zz] = event_extr_32_zz_neg

                                         cc_32_neg = np.where(np.all(nz_median_32_neighbors_neg[kk,:] == nz_ind_del_32_neg, axis=1))
                                         nz_ind_del_32_neg = np.delete(nz_ind_del_32_neg, cc_32_neg, axis=0)

                                 D1_32_neg = distance.cdist(np.array([nz_ind_zz_median_r_32_neg]), nz_ind_del_32_neg, 'euclidean')
                                 D1_sort_32_neg = np.argsort(D1_32_neg)


                                 if np.shape(D1_sort_32_neg)[1] > 0:
                                     del_ind_32_neg = D1_sort_32_neg[0, 0]
                                     nz_ind_zz_median_r_32_neg = nz_ind_del_32_neg[del_ind_32_neg, :]
                                     nz_ind_del_32_neg = np.delete(nz_ind_del_32_neg, del_ind_32_neg, axis=0)
                                 else:
                                     break

                    event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_32_neg

                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_32'''

                    event_extr_32_nz_ind_neg = np.nonzero(event_extr_32_neg)
                    event_extr_32_nz_c_neg = np.column_stack((event_extr_32_nz_ind_neg[0], event_extr_32_nz_ind_neg[1], event_extr_32_nz_ind_neg[2], event_extr_32_neg[event_extr_32_nz_ind_neg]))

                    event_extr_32_neg_yx = event_extr_32_nz_c_neg[:, :2]; event_extr_32_neg_p = event_extr_32_nz_c_neg[:, -1]
                    rle_32_p_neg, rle_32_p_bits_neg, rle_32_p_len_bits_neg = rle(event_extr_32_neg_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_32_neg_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_32_neg_yx[zzz, 0]) == huff_32_y1)
                        TOT_COMPR_BITS += len(huff_32_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_32_neg_yx[zzz, 1]) == huff_32_x1)
                        TOT_COMPR_BITS += len(huff_32_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_32_p_neg)[0]):
                        match_ind_p_neg = np.argwhere(int(rle_32_p_neg[yyy, 1]) == huff_32_p_neg1)
                        TOT_COMPR_BITS += len(huff_32_p_neg[match_ind_p_neg, 1])
            
            if ((int(b[2]) - int(b[0]) + 1) == 16) and ((int(b[3]) - int(b[1]) + 1) == 16):
                event_extr_16_pos = np.copy(event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_16_ccc_pos = np.count_nonzero(event_extr_16_pos)

                if event_extr_16_ccc_pos > 0:
                    for zz in range(Nbins_T):
                        event_extr_16_zz_pos = event_extr_16_pos[:, :, zz]
                        if np.count_nonzero(event_extr_16_zz_pos) > 0:
                             nz_ind_zz_16_pos = np.transpose(np.nonzero(event_extr_16_zz_pos))
                             event_extr_median_16_pos = geometric_median(nz_ind_zz_16_pos, 'auto')
                             D_16_pos = distance.cdist(np.array([event_extr_median_16_pos]), nz_ind_zz_16_pos, 'euclidean')
                             D_16_min_ind_pos = np.argmin(D_16_pos)
                             nz_ind_zz_median_r_16_pos = nz_ind_zz_16_pos[D_16_min_ind_pos, :] 
                             nz_ind_del_16_pos = np.delete(nz_ind_zz_16_pos, D_16_min_ind_pos, axis=0)
                             
                             while (len(nz_ind_del_16_pos) > 0) and (len(neighbors_16x16) > 0):
                                 nz_median_16_neighbors_pos = nz_ind_zz_median_r_16_pos + neighbors_16x16
                                 
                                 for kk in range(np.shape(neighbors_16x16)[0]):
                                     
                                     if (((nz_median_16_neighbors_pos[kk, 0] >= 0) and (nz_median_16_neighbors_pos[kk, 0] < block_size_16)) and\
                                         ((nz_median_16_neighbors_pos[kk, 1] >= 0) and (nz_median_16_neighbors_pos[kk, 1] < block_size_16))):
                                         
                                         event_extr_16_zz_pos[int(nz_median_16_neighbors_pos[kk,0]), int(nz_median_16_neighbors_pos[kk,1])] = 0
                                         event_extr_16_pos[:, :, zz] = event_extr_16_zz_pos
                                         
                                         cc_16_pos = np.where(np.all(nz_median_16_neighbors_pos[kk,:] == nz_ind_del_16_pos, axis=1))
                                         nz_ind_del_16_pos = np.delete(nz_ind_del_16_pos, cc_16_pos, axis=0)
                                 
                                 D1_16_pos = distance.cdist(np.array([nz_ind_zz_median_r_16_pos]), nz_ind_del_16_pos, 'euclidean')
                                 D1_sort_16_pos = np.argsort(D1_16_pos)
 
                                 
                                 if np.shape(D1_sort_16_pos)[1] > 0:
                                     del_ind_16_pos = D1_sort_16_pos[0, 0]
                                     nz_ind_zz_median_r_16_pos = nz_ind_del_16_pos[del_ind_16_pos, :] 
                                     nz_ind_del_16_pos = np.delete(nz_ind_del_16_pos, del_ind_16_pos, axis=0)
                                 else:
                                     break
            
                    event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_16_pos
                    
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_16_pos'''

                    event_extr_16_nz_ind_pos = np.nonzero(event_extr_16_pos)
                    event_extr_16_nz_c_pos = np.column_stack((event_extr_16_nz_ind_pos[0], event_extr_16_nz_ind_pos[1], event_extr_16_nz_ind_pos[2], event_extr_16_pos[event_extr_16_nz_ind_pos]))
                   
                    event_extr_16_pos_yx = event_extr_16_nz_c_pos[:, :2]; event_extr_16_pos_p = event_extr_16_nz_c_pos[:, -1]
                    rle_16_p_pos, rle_16_p_bits_pos, rle_16_p_len_bits_pos = rle(event_extr_16_pos_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_16_pos_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_16_pos_yx[zzz, 0]) == huff_16_y1)
                        TOT_COMPR_BITS += len(huff_16_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_16_pos_yx[zzz, 1]) == huff_16_x1)
                        TOT_COMPR_BITS += len(huff_16_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_16_p_pos)[0]):
                        match_ind_p_pos = np.argwhere(int(rle_16_p_pos[yyy, 1]) == huff_16_p_pos1)
                        TOT_COMPR_BITS += len(huff_16_p_pos[match_ind_p_pos, 1])

            if ((int(b[2]) - int(b[0]) + 1) == 16) and ((int(b[3]) - int(b[1]) + 1) == 16):
                event_extr_16_neg = np.copy(event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_16_ccc_neg = np.count_nonzero(event_extr_16_neg)

                if event_extr_16_ccc_neg > 0:
                    for zz in range(Nbins_T):
                        event_extr_16_zz_neg = event_extr_16_neg[:, :, zz]
                        if np.count_nonzero(event_extr_16_zz_neg) > 0:
                             nz_ind_zz_16_neg = np.transpose(np.nonzero(event_extr_16_zz_neg))
                             event_extr_median_16_neg = geometric_median(nz_ind_zz_16_neg, 'auto')
                             D_16_neg = distance.cdist(np.array([event_extr_median_16_neg]), nz_ind_zz_16_neg, 'euclidean')
                             D_16_min_ind_neg = np.argmin(D_16_neg)
                             nz_ind_zz_median_r_16_neg = nz_ind_zz_16_neg[D_16_min_ind_neg, :]
                             nz_ind_del_16_neg = np.delete(nz_ind_zz_16_neg, D_16_min_ind_neg, axis=0)

                             while (len(nz_ind_del_16_neg) > 0) and (len(neighbors_16x16) > 0):
                                 nz_median_16_neighbors_neg = nz_ind_zz_median_r_16_neg + neighbors_16x16

                                 for kk in range(np.shape(neighbors_16x16)[0]):

                                     if (((nz_median_16_neighbors_neg[kk, 0] >= 0) and (nz_median_16_neighbors_neg[kk, 0] < block_size_16)) and\
                                         ((nz_median_16_neighbors_neg[kk, 1] >= 0) and (nz_median_16_neighbors_neg[kk, 1] < block_size_16))):

                                         event_extr_16_zz_neg[int(nz_median_16_neighbors_neg[kk,0]), int(nz_median_16_neighbors_neg[kk,1])] = 0
                                         event_extr_16_neg[:, :, zz] = event_extr_16_zz_neg

                                         cc_16_neg = np.where(np.all(nz_median_16_neighbors_neg[kk,:] == nz_ind_del_16_neg, axis=1))
                                         nz_ind_del_16_neg = np.delete(nz_ind_del_16_neg, cc_16_neg, axis=0)

                                 D1_16_neg = distance.cdist(np.array([nz_ind_zz_median_r_16_neg]), nz_ind_del_16_neg, 'euclidean')
                                 D1_sort_16_neg = np.argsort(D1_16_neg)


                                 if np.shape(D1_sort_16_neg)[1] > 0:
                                     del_ind_16_neg = D1_sort_16_neg[0, 0]
                                     nz_ind_zz_median_r_16_neg = nz_ind_del_16_neg[del_ind_16_neg, :]
                                     nz_ind_del_16_neg = np.delete(nz_ind_del_16_neg, del_ind_16_neg, axis=0)
                                 else:
                                     break


                    event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_16_neg

                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_16_pos'''

                    event_extr_16_nz_ind_neg = np.nonzero(event_extr_16_neg)
                    event_extr_16_nz_c_neg = np.column_stack((event_extr_16_nz_ind_neg[0], event_extr_16_nz_ind_neg[1], event_extr_16_nz_ind_neg[2], event_extr_16_neg[event_extr_16_nz_ind_neg]))
                   
                    event_extr_16_neg_yx = event_extr_16_nz_c_neg[:, :2]; event_extr_16_neg_p = event_extr_16_nz_c_neg[:, -1]
                    rle_16_p_neg, rle_16_p_bits_neg, rle_16_p_len_bits_neg = rle(event_extr_16_neg_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_16_neg_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_16_neg_yx[zzz, 0]) == huff_16_y1)
                        TOT_COMPR_BITS += len(huff_16_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_16_neg_yx[zzz, 1]) == huff_16_x1)
                        TOT_COMPR_BITS += len(huff_16_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_16_p_neg)[0]):
                        match_ind_p_neg = np.argwhere(int(rle_16_p_neg[yyy, 1]) == huff_16_p_neg1)
                        TOT_COMPR_BITS += len(huff_16_p_neg[match_ind_p_neg, 1])


            if ((int(b[2]) - int(b[0]) + 1) == 8) and ((int(b[3]) - int(b[1]) + 1) == 8):
                event_extr_8_pos = np.copy(event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_8_ccc_pos = np.count_nonzero(event_extr_8_pos)

                if event_extr_8_ccc_pos > 0:
                    for zz in range(Nbins_T): #this is for each quantized time 
                        event_extr_8_zz_pos = event_extr_8_pos[:, :, zz]
                        if np.count_nonzero(event_extr_8_zz_pos) > 0:
                            nz_ind_zz_8_pos = np.transpose(np.nonzero(event_extr_8_zz_pos))
                            event_extr_median_8_pos = geometric_median(nz_ind_zz_8_pos, 'auto')
                            D_8_pos = distance.cdist(np.array([event_extr_median_8_pos]), nz_ind_zz_8_pos, 'euclidean')
                            D_8_min_ind_pos = np.argmin(D_8_pos)
                            nz_ind_zz_median_r_8_pos = nz_ind_zz_8_pos[D_8_min_ind_pos, :] 
                            nz_ind_del_8_pos = np.delete(nz_ind_zz_8_pos, D_8_min_ind_pos, axis=0)

                            while (len(nz_ind_del_8_pos) > 0) and (len(neighbors_8x8) > 0):
                                nz_median_8_neighbors_pos = nz_ind_zz_median_r_8_pos + neighbors_8x8

                                for kk in range(np.shape(neighbors_8x8)[0]):

                                    if (((nz_median_8_neighbors_pos[kk, 0] >= 0) and (nz_median_8_neighbors_pos[kk, 0] < block_size_8)) and\
                                        ((nz_median_8_neighbors_pos[kk, 1] >= 0) and (nz_median_8_neighbors_pos[kk, 1] < block_size_8))):

                                        event_extr_8_zz_pos[int(nz_median_8_neighbors_pos[kk,0]), int(nz_median_8_neighbors_pos[kk,1])] = 0
                                        event_extr_8_pos[:, :, zz] = event_extr_8_zz_pos
                                        
                                        cc_8_pos = np.where(np.all(nz_median_8_neighbors_pos[kk,:] == nz_ind_del_8_pos, axis=1))
                                        nz_ind_del_8_pos = np.delete(nz_ind_del_8_pos, cc_8_pos, axis=0)

                                D1_8_pos = distance.cdist(np.array([nz_ind_zz_median_r_8_pos]), nz_ind_del_8_pos, 'euclidean')
                                D1_sort_8_pos = np.argsort(D1_8_pos)


                                if np.shape(D1_sort_8_pos)[1] > 0:
                                    del_ind_8_pos = D1_sort_8_pos[0, 0]
                                    nz_ind_zz_median_r_8_pos = nz_ind_del_8_pos[del_ind_8_pos, :]
                                    nz_ind_del_8_pos = np.delete(nz_ind_del_8_pos, del_ind_8_pos, axis=0)
                                else:
                                    break
                    
                    event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_8_pos
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_8'''
                    event_extr_8_nz_ind_pos = np.nonzero(event_extr_8_pos)
                    event_extr_8_nz_c_pos = np.column_stack((event_extr_8_nz_ind_pos[0], event_extr_8_nz_ind_pos[1], event_extr_8_nz_ind_pos[2], event_extr_8_pos[event_extr_8_nz_ind_pos]))
                    
                    event_extr_8_pos_yx = event_extr_8_nz_c_pos[:, :2]; event_extr_8_pos_p = event_extr_8_nz_c_pos[:, -1]
                    rle_8_p_pos, rle_8_p_bits_pos, rle_8_p_len_bits_pos = rle(event_extr_8_pos_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_8_pos_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_8_pos_yx[zzz, 0]) == huff_8_y1)
                        TOT_COMPR_BITS += len(huff_8_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_8_pos_yx[zzz, 1]) == huff_8_x1)
                        TOT_COMPR_BITS += len(huff_8_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_8_p_pos)[0]):
                        match_ind_p_pos = np.argwhere(int(rle_8_p_pos[yyy, 1]) == huff_8_p_pos1)
                        TOT_COMPR_BITS += len(huff_8_p_pos[match_ind_p_pos, 1])


            if ((int(b[2]) - int(b[0]) + 1) == 8) and ((int(b[3]) - int(b[1]) + 1) == 8):
                event_extr_8_neg = np.copy(event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_8_ccc_neg = np.count_nonzero(event_extr_8_neg)
 
                if event_extr_8_ccc_neg > 0:
                    for zz in range(Nbins_T): #this is for each quantized time 
                        event_extr_8_zz_neg = event_extr_8_neg[:, :, zz]
                        if np.count_nonzero(event_extr_8_zz_neg) > 0:
                            nz_ind_zz_8_neg = np.transpose(np.nonzero(event_extr_8_zz_neg))
                            event_extr_median_8_neg = geometric_median(nz_ind_zz_8_neg, 'auto')
                            D_8_neg = distance.cdist(np.array([event_extr_median_8_neg]), nz_ind_zz_8_neg, 'euclidean')
                            D_8_min_ind_neg = np.argmin(D_8_neg)
                            nz_ind_zz_median_r_8_neg = nz_ind_zz_8_neg[D_8_min_ind_neg, :]
                            nz_ind_del_8_neg = np.delete(nz_ind_zz_8_neg, D_8_min_ind_neg, axis=0)

                            while (len(nz_ind_del_8_neg) > 0) and (len(neighbors_8x8) > 0):
                                nz_median_8_neighbors_neg = nz_ind_zz_median_r_8_neg + neighbors_8x8
 
                                for kk in range(np.shape(neighbors_8x8)[0]):

                                    if (((nz_median_8_neighbors_neg[kk, 0] >= 0) and (nz_median_8_neighbors_neg[kk, 0] < block_size_8)) and\
                                        ((nz_median_8_neighbors_neg[kk, 1] >= 0) and (nz_median_8_neighbors_neg[kk, 1] < block_size_8))):

                                        event_extr_8_zz_neg[int(nz_median_8_neighbors_neg[kk,0]), int(nz_median_8_neighbors_neg[kk,1])] = 0
                                        event_extr_8_neg[:, :, zz] = event_extr_8_zz_neg
 
                                        cc_8_neg = np.where(np.all(nz_median_8_neighbors_neg[kk,:] == nz_ind_del_8_neg, axis=1))
                                        nz_ind_del_8_neg = np.delete(nz_ind_del_8_neg, cc_8_neg, axis=0)
 
                                D1_8_neg = distance.cdist(np.array([nz_ind_zz_median_r_8_neg]), nz_ind_del_8_neg, 'euclidean')
                                D1_sort_8_neg = np.argsort(D1_8_neg)
 
 
                                if np.shape(D1_sort_8_neg)[1] > 0:
                                    del_ind_8_neg = D1_sort_8_neg[0, 0]
                                    nz_ind_zz_median_r_8_neg = nz_ind_del_8_neg[del_ind_8_neg, :]
                                    nz_ind_del_8_neg = np.delete(nz_ind_del_8_neg, del_ind_8_neg, axis=0)
                                else:
                                    break

                    event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_8_neg
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_8'''
                    event_extr_8_nz_ind_neg = np.nonzero(event_extr_8_neg)
                    event_extr_8_nz_c_neg = np.column_stack((event_extr_8_nz_ind_neg[0], event_extr_8_nz_ind_neg[1], event_extr_8_nz_ind_neg[2], event_extr_8_neg[event_extr_8_nz_ind_neg]))                   
                    event_extr_8_neg_yx = event_extr_8_nz_c_neg[:, :2]; event_extr_8_neg_p = event_extr_8_nz_c_neg[:, -1]
                    rle_8_p_neg, rle_8_p_bits_neg, rle_8_p_len_bits_neg = rle(event_extr_8_neg_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_8_neg_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_8_neg_yx[zzz, 0]) == huff_8_y1)
                        TOT_COMPR_BITS += len(huff_8_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_8_neg_yx[zzz, 1]) == huff_8_x1)
                        TOT_COMPR_BITS += len(huff_8_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_8_p_neg)[0]):
                        match_ind_p_neg = np.argwhere(int(rle_8_p_neg[yyy, 1]) == huff_8_p_neg1)
                        TOT_COMPR_BITS += len(huff_8_p_neg[match_ind_p_neg, 1])

            if ((int(b[2]) - int(b[0]) + 1) == 4) and ((int(b[3]) - int(b[1]) + 1) == 4):
                event_extr_4_pos = np.copy(event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_4_ccc_pos = np.count_nonzero(event_extr_4_pos)

                if event_extr_4_ccc_pos > 0:
                    for zz in range(Nbins_T): #this is for each quantized time 
                        event_extr_4_zz_pos = event_extr_4_pos[:, :, zz]
                        if np.count_nonzero(event_extr_4_zz_pos) > 0:
                            nz_ind_zz_pos = np.transpose(np.nonzero(event_extr_4_zz_pos))
                            nz_ind_zz_median_pos = geometric_median(nz_ind_zz_pos, 'auto')
                            D_4_pos = distance.cdist(np.array([nz_ind_zz_median_pos]), nz_ind_zz_pos, 'euclidean')
                            D_4_min_ind_pos = np.argmin(D_4_pos)
                            nz_ind_zz_median_r_pos = nz_ind_zz_pos[D_4_min_ind_pos, :]
                            nz_ind_del_pos = np.delete(nz_ind_zz_pos, D_4_min_ind_pos, axis=0)

                            while (len(nz_ind_del_pos) > 0) and (len(neighbors_4x4) > 0):
                                nz_median_4_neighbors_pos = nz_ind_zz_median_r_pos + neighbors_4x4

                                for kk in range(np.shape(neighbors_4x4)[0]):

                                    if (((nz_median_4_neighbors_pos[kk, 0] >= 0) and (nz_median_4_neighbors_pos[kk, 0] < block_size_4)) and\
                                        ((nz_median_4_neighbors_pos[kk, 1] >= 0) and (nz_median_4_neighbors_pos[kk, 1] < block_size_4))):

                                        event_extr_4_zz_pos[int(nz_median_4_neighbors_pos[kk,0]), int(nz_median_4_neighbors_pos[kk,1])] = 0
                                        event_extr_4_pos[:, :, zz] = event_extr_4_zz_pos

                                        cc_4_pos = np.where(np.all(nz_median_4_neighbors_pos[kk,:] == nz_ind_del_pos, axis=1))
                                        nz_ind_del_pos = np.delete(nz_ind_del_pos, cc_4_pos, axis=0)

                                D1_pos = distance.cdist(np.array([nz_ind_zz_median_r_pos]), nz_ind_del_pos, 'euclidean')
                                D1_sort_pos = np.argsort(D1_pos)


                                if np.shape(D1_sort_pos)[1] > 0:
                                    del_ind_pos = D1_sort_pos[0, 0]
                                    nz_ind_zz_median_r_pos = nz_ind_del_pos[del_ind_pos, :]
                                    nz_ind_del_pos = np.delete(nz_ind_del_pos, del_ind_pos, axis=0)
                                else:
                                    break
                    
                    event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_4_pos
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_4'''
                    event_extr_4_nz_ind_pos = np.nonzero(event_extr_4_pos)
                    event_extr_4_nz_c_pos = np.column_stack((event_extr_4_nz_ind_pos[0], event_extr_4_nz_ind_pos[1], event_extr_4_nz_ind_pos[2], event_extr_4_pos[event_extr_4_nz_ind_pos]))
                    event_extr_4_pos_yx = event_extr_4_nz_c_pos[:, :2]; event_extr_4_pos_p = event_extr_4_nz_c_pos[:, -1]
                    rle_4_p_pos, rle_4_p_bits_pos, rle_4_p_len_bits_pos = rle(event_extr_4_pos_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_4_pos_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_4_pos_yx[zzz, 0]) == huff_4_y1)
                        TOT_COMPR_BITS += len(huff_4_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_4_pos_yx[zzz, 1]) == huff_4_x1)
                        TOT_COMPR_BITS += len(huff_4_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_4_p_pos)[0]):
                        match_ind_p_pos = np.argwhere(int(rle_4_p_pos[yyy, 1]) == huff_4_p_pos1)
                        TOT_COMPR_BITS += len(huff_4_p_pos[match_ind_p_pos, 1]) 

            if ((int(b[2]) - int(b[0]) + 1) == 4) and ((int(b[3]) - int(b[1]) + 1) == 4):
                event_extr_4_neg = np.copy(event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_4_ccc_neg = np.count_nonzero(event_extr_4_neg)

                if event_extr_4_ccc_neg > 0:
                    for zz in range(Nbins_T): #this is for each quantized time
                        event_extr_4_zz_neg = event_extr_4_neg[:, :, zz]
                        if np.count_nonzero(event_extr_4_zz_neg) > 0:
                            nz_ind_zz_neg = np.transpose(np.nonzero(event_extr_4_zz_neg))
                            nz_ind_zz_median_neg = geometric_median(nz_ind_zz_neg, 'auto')
                            D_4_neg = distance.cdist(np.array([nz_ind_zz_median_neg]), nz_ind_zz_neg, 'euclidean')
                            D_4_min_ind_neg = np.argmin(D_4_neg)
                            nz_ind_zz_median_r_neg = nz_ind_zz_neg[D_4_min_ind_neg, :]
                            nz_ind_del_neg = np.delete(nz_ind_zz_neg, D_4_min_ind_neg, axis=0)
                            
                            while (len(nz_ind_del_neg) > 0) and (len(neighbors_4x4) > 0):
                                nz_median_4_neighbors_neg = nz_ind_zz_median_r_neg + neighbors_4x4

                                for kk in range(np.shape(neighbors_4x4)[0]):

                                    if (((nz_median_4_neighbors_neg[kk, 0] >= 0) and (nz_median_4_neighbors_neg[kk, 0] < block_size_4)) and\
                                        ((nz_median_4_neighbors_neg[kk, 1] >= 0) and (nz_median_4_neighbors_neg[kk, 1] < block_size_4))):

                                        event_extr_4_zz_neg[int(nz_median_4_neighbors_neg[kk,0]), int(nz_median_4_neighbors_neg[kk,1])] = 0
                                        event_extr_4_neg[:, :, zz] = event_extr_4_zz_neg

                                        cc_4_neg = np.where(np.all(nz_median_4_neighbors_neg[kk,:] == nz_ind_del_neg, axis=1))
                                        nz_ind_del_neg = np.delete(nz_ind_del_neg, cc_4_neg, axis=0)

                                D1_neg = distance.cdist(np.array([nz_ind_zz_median_r_neg]), nz_ind_del_neg, 'euclidean')
                                D1_sort_neg = np.argsort(D1_neg)


                                if np.shape(D1_sort_neg)[1] > 0:
                                    del_ind_neg = D1_sort_neg[0, 0]
                                    nz_ind_zz_median_r_neg = nz_ind_del_neg[del_ind_neg, :]
                                    nz_ind_del_neg = np.delete(nz_ind_del_neg, del_ind_neg, axis=0)
                                else:
                                    break

                    event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_4_neg
                    '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_4'''
                    event_extr_4_nz_ind_neg = np.nonzero(event_extr_4_neg)
                    event_extr_4_nz_c_neg = np.column_stack((event_extr_4_nz_ind_neg[0], event_extr_4_nz_ind_neg[1], event_extr_4_nz_ind_neg[2], event_extr_4_neg[event_extr_4_nz_ind_neg]))
                    event_extr_4_neg_yx = event_extr_4_nz_c_neg[:, :2]; event_extr_4_neg_p = event_extr_4_nz_c_neg[:, -1]
                    rle_4_p_neg, rle_4_p_bits_neg, rle_4_p_len_bits_neg = rle(event_extr_4_neg_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_4_neg_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_4_neg_yx[zzz, 0]) == huff_4_y1)
                        TOT_COMPR_BITS += len(huff_4_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_4_neg_yx[zzz, 1]) == huff_4_x1)
                        TOT_COMPR_BITS += len(huff_4_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_4_p_neg)[0]):
                        match_ind_p_neg = np.argwhere(int(rle_4_p_neg[yyy, 1]) == huff_4_p_neg1)
                        TOT_COMPR_BITS += len(huff_4_p_neg[match_ind_p_neg, 1])


            if r_2 == 1:
                r_2_step = 2
            elif r_2 > 1:
                r_2_step = 1
            else:
                r_2_step = 0

            if ((int(b[2]) - int(b[0]) + 1) == 2) and ((int(b[3]) - int(b[1]) + 1) == 2):
                event_extr_2_pos = np.copy(event_3D_pos[int(b[0]) - 1: int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_2_ccc_pos = np.count_nonzero(event_extr_2_pos)

                if event_extr_2_ccc_pos > 0:
                    for zz in range(Nbins_T):
                        event_extr_2_zz_pos = event_extr_2_pos[:, :, zz]
                        if (np.count_nonzero(event_extr_2_zz_pos) > 0) and (r_2_step >= 1):
                            nz_ind_zz_pos = np.transpose(np.nonzero(event_extr_2_zz_pos))
                            lennn_pos = np.shape(nz_ind_zz_pos)[0]
                            if (lennn_pos > 1) and (comp2==1):
                                for kk in range(1, lennn_pos, r_2_step):
                                    event_extr_2_zz_pos[int(nz_ind_zz_pos[kk, 0]), int(nz_ind_zz_pos[kk, 1])] = 0
                                    
                        event_extr_2_pos[:, :, zz] = event_extr_2_zz_pos


                event_3D_pos[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_2_pos
                '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_2'''
                event_extr_2_pos = np.copy(event_3D_pos[int(b[0]) - 1: int(b[2]), int(b[1])-1:int(b[3]), :])   
                event_extr_2_ccc_pos = np.count_nonzero(event_extr_2_pos)
                
                if event_extr_2_ccc_pos > 0:
                    event_extr_2_nz_ind_pos = np.nonzero(event_extr_2_pos)
                    event_extr_2_nz_c_pos = np.column_stack((event_extr_2_nz_ind_pos[0], event_extr_2_nz_ind_pos[1], event_extr_2_nz_ind_pos[2], event_extr_2_pos[event_extr_2_nz_ind_pos]))
                    event_extr_2_pos_yx = event_extr_2_nz_c_pos[:, :2]; event_extr_2_pos_p = event_extr_2_nz_c_pos[:, -1]
                    rle_2_p_pos, rle_2_p_bits_pos, rle_2_p_len_bits_pos = rle(event_extr_2_pos_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_2_pos_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_2_pos_yx[zzz, 0]) == huff_2_y1)
                        TOT_COMPR_BITS += len(huff_2_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_2_pos_yx[zzz, 1]) == huff_2_x1)
                        TOT_COMPR_BITS += len(huff_2_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_2_p_pos)[0]):
                        match_ind_p_pos = np.argwhere(int(rle_2_p_pos[yyy, 1]) == huff_2_p_pos1)
                        TOT_COMPR_BITS += len(huff_2_p_pos[match_ind_p_pos, 1])


            if ((int(b[2]) - int(b[0]) + 1) == 2) and ((int(b[3]) - int(b[1]) + 1) == 2):
                event_extr_2_neg = np.copy(event_3D_neg[int(b[0]) - 1: int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_2_ccc_neg = np.count_nonzero(event_extr_2_neg)

                if event_extr_2_ccc_neg > 0:
                    for zz in range(Nbins_T):
                        event_extr_2_zz_neg = event_extr_2_neg[:, :, zz]
                        if (np.count_nonzero(event_extr_2_zz_neg) > 0) and (r_2_step >= 1):
                            nz_ind_zz_neg = np.transpose(np.nonzero(event_extr_2_zz_neg))
                            lennn_neg = np.shape(nz_ind_zz_neg)[0]
                            if (lennn_neg > 1) and (comp2==1):
                                for kk in range(1, lennn_neg, r_2_step):
                                    event_extr_2_zz_neg[int(nz_ind_zz_neg[kk, 0]), int(nz_ind_zz_neg[kk, 1])] = 0

                        event_extr_2_neg[:, :, zz] = event_extr_2_zz_neg


                event_3D_neg[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_2_neg
                '''event_3D_new[int(b[0]) - 1:int(b[2]), int(b[1]) - 1:int(b[3]), :] = event_extr_2'''
                event_extr_2_neg = np.copy(event_3D_neg[int(b[0]) - 1: int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_2_ccc_neg = np.count_nonzero(event_extr_2_neg)

                if event_extr_2_ccc_neg > 0:
                    event_extr_2_nz_ind_neg = np.nonzero(event_extr_2_neg)
                    event_extr_2_nz_c_neg = np.column_stack((event_extr_2_nz_ind_neg[0], event_extr_2_nz_ind_neg[1], event_extr_2_nz_ind_neg[2], event_extr_2_neg[event_extr_2_nz_ind_neg]))
                    event_extr_2_neg_yx = event_extr_2_nz_c_neg[:, :2]; event_extr_2_neg_p = event_extr_2_nz_c_neg[:, -1]
                    rle_2_p_neg, rle_2_p_bits_neg, rle_2_p_len_bits_neg = rle(event_extr_2_neg_p, binary = False)
                         
                    for zzz in range(np.shape(event_extr_2_neg_yx)[0]):
                        match_ind_y = np.argwhere(int(event_extr_2_neg_yx[zzz, 0]) == huff_2_y1)
                        TOT_COMPR_BITS += len(huff_2_y[match_ind_y, 1])
                        match_ind_x = np.argwhere(int(event_extr_2_neg_yx[zzz, 1]) == huff_2_x1)
                        TOT_COMPR_BITS += len(huff_2_x[match_ind_x, 1])
                             
                    for yyy in range(np.shape(rle_2_p_neg)[0]):
                        match_ind_p_neg = np.argwhere(int(rle_2_p_neg[yyy, 1]) == huff_2_p_neg1)
                        TOT_COMPR_BITS += len(huff_2_p_neg[match_ind_p_neg, 1])

           

            if ((int(b[2]) - int(b[0]) + 1) == 1) and ((int(b[3]) - int(b[1]) + 1) == 1):
                # extract events in these regions in Nbins_T plane
                event_extr_1_pos = np.copy(event_3D_pos[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_1_ccc_pos = np.count_nonzero(event_extr_1_pos)                        

                if event_extr_1_ccc_pos > 0: # see if there is any events at all
                    event_extr_1_nz_ind_pos = np.nonzero(event_extr_1_pos)
          
                    # append position (x,y,t_bin) and polarity
                    event_extr_1_nz_c_pos = np.column_stack((event_extr_1_nz_ind_pos[0], event_extr_1_nz_ind_pos[1], event_extr_1_nz_ind_pos[2], event_extr_1_pos[event_extr_1_nz_ind_pos]))
                    chip_host_events_1x1_pos.append(event_extr_1_nz_c_pos)  
    

            if ((int(b[2]) - int(b[0]) + 1) == 1) and ((int(b[3]) - int(b[1]) + 1) == 1):
                # extract events in these regions in Nbins_T plane
                event_extr_1_neg = np.copy(event_3D_neg[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :])
                event_extr_1_ccc_neg = np.count_nonzero(event_extr_1_neg)
 
                if event_extr_1_ccc_neg > 0: # see if there is any events at all
                    event_extr_1_nz_ind_neg = np.nonzero(event_extr_1_neg)

                    # append position (x,y,t_bin) and polarity
                    event_extr_1_nz_c_neg = np.column_stack((event_extr_1_nz_ind_neg[0], event_extr_1_nz_ind_neg[1], event_extr_1_nz_ind_neg[2], event_extr_1_neg[event_extr_1_nz_ind_neg]))
                    chip_host_events_1x1_neg.append(event_extr_1_nz_c_neg)

        else:

            '''this corresponds to skip cases '''
            '''copy from previous event frame in the host (transient cases to be considered later) '''
            #event_3D[int(b[0])-1:int(b[2]), int(b[1])-1:int(b[3]), :] = 0
    
    if len(chip_host_events_1x1_pos) > 0:
        chip_ev_1x1_pos = np.concatenate(chip_host_events_1x1_pos)
        ev_extr_1_p_pos = chip_ev_1x1_pos[:, 3]
        rle_1_p_pos, rle_1_p_bits_pos, rle_1_p_len_bits_pos = rle(ev_extr_1_p_pos, binary = False)
 
        for yyy in range(np.shape(rle_1_p_pos)[0]):
            match_ind_p_pos = np.argwhere(int(rle_1_p_pos[yyy, 1]) == huff_1_p_pos1)
            TOT_COMPR_BITS += len(huff_1_p_pos[match_ind_p_pos, 1])
    
    if len(chip_host_events_1x1_neg) > 0:
        chip_ev_1x1_neg = np.concatenate(chip_host_events_1x1_neg)
        ev_extr_1_p_neg = chip_ev_1x1_neg[:, 3]
        rle_1_p_neg, rle_1_p_bits_neg, rle_1_p_len_bits_neg = rle(np.abs(ev_extr_1_p_neg), binary = False)
        
        for yyy in range(np.shape(rle_1_p_neg)[0]):
            match_ind_p_neg = np.argwhere(int(rle_1_p_neg[yyy, 1]) == huff_1_p_neg1)
            TOT_COMPR_BITS += len(huff_1_p_neg[match_ind_p_neg, 1])

    TOT_COMPR_BITS += (Nbins_T*len(np.binary_repr(Nbins_T - 1)))
    
    td_image_comp = np.zeros((256, 256), dtype=np.uint8)
    td_image_comp = np.sum(np.abs(event_3D_pos), axis=2) + np.sum(np.abs(event_3D_neg), axis=2)
    print('td_image_comp max and min', np.max(td_image_comp), np.min(td_image_comp))
    MSE = np.mean((td_image_org - td_image_comp)**2)
    PSNR = 20 * np.log10(Nbins_T/np.sqrt(MSE))
    SSIM = measure.compare_ssim(td_image_org, td_image_comp, data_range=np.max(td_image_org) - np.min(td_image_org))
       

    #event_3D = np.copy(event_3D_new)
    #print('total_events', total_events)
    #print('compression ratio', np.count_nonzero(event_3D) / total_events)
    print('TOTAL COMPRESSED BITS', TOT_COMPR_BITS)
    print('TOTAL UNCOMPRESSED BITS (64 bit repr)', BIT_STD1)
    print('compressed positive events', np.count_nonzero(event_3D_pos)); print('compressed negative events', np.count_nonzero(event_3D_neg))
    
    Nbins_T_ind = np.linspace(0, Nbins_T, Nbins_T+1)
    T_error = 0
    Tot_org_ev = np.shape(events_org)[0]
    for jj in range(Tot_org_ev):
        ev_act_time = events_org[jj,2]
        ev_quant_time = np.digitize(ev_act_time, T_disc) - 1
        ev_quant_time_act = np.argwhere(ev_quant_time==Nbins_T_ind)
        ev_quant_time_disc = np.asscalar(T_disc[ev_quant_time_act])
        #print('ev_quant_time', ev_quant_time)
        #print('ev act time, ev_quant_time_disc', ev_act_time, ev_quant_time_disc)

        if (event_3D_pos[int(events_org[jj,0]) + 38, int(events_org[jj,1])+33, int(ev_quant_time)] != 0) or (event_3D_neg[int(events_org[jj,0]) + 38, int(events_org[jj,1])+33, int(ev_quant_time)] != 0):
            T_diff = (ev_act_time - ev_quant_time_disc)**2
        else:
            T_diff = ev_act_time**2

        T_error += T_diff

    T_error_sqrt = np.sqrt(T_error)
    T_error_ret = T_error_sqrt
    total_events = np.count_nonzero(event_3D_pos) + np.count_nonzero(event_3D_neg)
    '''
    event_3D_c_ind = np.transpose(np.nonzero(event_3D)); event_3D_sh = np.shape(event_3D_c_ind)[0]; event_3D_4 = np.zeros(shape=(event_3D_sh, 4))
    event_3D_4[:, :3] = np.copy(event_3D_c_ind)

    for ii in range(event_3D_sh):
        event_3D_4[ii, 3] = event_3D[event_3D_c_ind[ii, 0], event_3D_c_ind[ii, 1], event_3D_c_ind[ii, 2]]
   
    event_3D_4_t = event_3D_4[:, 2]
    event_3D_4_t = np.piecewise(event_3D_4_t, [event_3D_4_t==0, event_3D_4_t==1, event_3D_4_t==2, event_3D_4_t==3, event_3D_4_t==4, event_3D_4_t==5,\
    event_3D_4_t==6, event_3D_4_t==7], [0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175])
    event_3D_4[:, 2] = np.copy(event_3D_4_t)

    #print('event_3D_4', event_3D_4)
    #print('events-'+str(k123))
    #np.savetxt('GEF_dataset_results/Indoor3_trial/events-'+str(k123)+'.txt', event_3D_4)
    '''

    '''chip host events are the compressed events sent from the chip to the host'''
    return event_3D_pos, event_3D_neg, TOTAL_EVENTS_ORG, ppp, total_events, TOT_COMPR_BITS, BIT_STD1, PSNR, SSIM, T_error_ret

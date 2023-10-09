from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import xml.etree.ElementTree as ET
import math as ma
import shutil
import time
import copy
import pylab
import imageio
import skimage.io as io
import concurrent.futures
from skimage import transform,io
from qt_orient import *
from qt_segment import *
from qt_sets import *
from qt_squares_bboxes import *
from qt_viterbi_segm import *
from qt_events import *
from qt_reconstruct_b import *
from qt_event_reconstruct import *
from qt_values import *
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy
from natsort import natsorted
from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
import json
from tqdm import tqdm
from natsort import natsorted


# Actual ReImagine Code Starts

def next_lambda_bezier(Rm, R0, D0, lambda0, R1, D1, lambda1):
    d0 = -1/lambda0
    d1 = -1/lambda1
    a0 = R0 - (d0*D0)
    a1 = R1 - (d1*D1)
    den = d1 - d0
    Db = (a0 - a1)/den
    Rb = ((d1*a0) - (d0*a1))/den
    dx = (R1 - Rb) - (Rb - R0)

    if dx == 0:
        umax = (Rm - R0)/(2*(Rb - R0))
    else:
        A = (R0 - Rb)/dx
        A2 = ma.pow(A,2)
        T1 = (A2 + ((Rm - R0)/dx))
        B = ma.pow(T1,0.5)
        umax = A + B
        if umax > 1 or umax < 0:
            umax = A - B

    dmax = (((umax - 1)*R0) + ((1 - (2*umax))*Rb) + (umax*R1))/(((umax-1)*D0) + (((1 -(2*umax))*Db) + (umax*D1)))
    lambda3 = -ma.pow(dmax,-1)
    return lambda3

def qt_viterbi_segm_init(lambdaa):
    [tree, gxf_min, total_rate] = qt_viterbi_segm(f2,f3,N,n0,o,a,f,t,segm_track_p,priorities,lambdaa, area)
    return [tree, gxf_min, total_rate]

folder_p = sys.argv[1]
print('folder_p', folder_p)

# Directory where distorted video files are stored
rootdir_det = "/home/srutarshi/Desktop/nDistortion_mult/Option2_fin_FasterRCNN-MultiClassKP/BMVC_2020/GEF_dataset_results/Sept18/100kbps_r4_1_corr/"+folder_p+"/Tbins_16a_Git"

if not os.path.exists(rootdir_det):
    os.makedirs(rootdir_det)

rootdir_d_fi = "SpatioTemp_comp"
rootd_det = os.path.join(rootdir_det, rootdir_d_fi)
if not os.path.exists(rootd_det):
    os.makedirs(rootd_det)


root_ev_path = rootd_det + "/ev_c_files"
if not os.path.exists(root_ev_path):
    os.makedirs(root_ev_path)

# construct the argument parse and parse the arguments
COLORS = np.random.uniform(0, 255, size=(21, 3))
new_rows = 256 #rows in new image
new_cols = 256 #cols in new image


out_rate = np.zeros((1,3))
opt_256 = 256
#opt_512 = 512
#opt_1024 = 1024

#nf < NF for partial experiments, nf = NF for full experiments
#nf = np.clip(nf,0,900) # number of frames clipped to 900 frames
visualize = 1 # Take this as the input from user at a later stage

N = int(ma.log(256, 2))
n0 = 0
Nl = N - n0 + 1
Max_Rate = 256*256*8
int_fact = 1.0
Rm = 2000*int_fact  # 2000 is 0.1 Mbits/s or 100 kbits/s 
Re = 2000*(1-int_fact) # 10000 is 0.5 Mbits/s or 500 kbits/s

lambda0 = 0.1
lambda1 = 10000

'''
Intensity stuff goes here
'''
print("Loading intensity data")
intensity_dir = "/home/srutarshi/Desktop/nDistortion_mult/Option2_fin_FasterRCNN-MultiClassKP/BMVC_2020/GEF_dataset/"+folder_p+"/RGB_frame"
int_filenames = []
for root, dirs, files in os.walk(intensity_dir):
    for f in files:
        int_filenames.append(root+"/"+f)

int_filenames_sorted = natsorted(int_filenames)

nf = len(int_filenames_sorted)
# extract only 1024x1024 region
#xmin = 208; xmax = 1232; ymin = 248; ymax = 1272

# extract only 512x512 region
#xmin = 464; xmax = 976; ymin = 504; ymax = 1016

'''f1 = cv2.imread(int_filenames_sorted[0], 0)'''
f2 = cv2.imread(int_filenames_sorted[0], 0)
print('f2 shape', np.shape(f2))

'''f1 = cv2.resize(f1, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)'''
f2 = cv2.resize(f2, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
'''print('f1 shape', np.shape(f1), np.max(f1), np.min(f1), type(f1))'''
print('f2 shape', np.shape(f2), np.max(f2), np.min(f2), type(f2))
#cv2.imwrite('f1_org.jpg', f1)
#blabla

f2_256 = np.zeros(shape=(256,256), dtype=np.uint8)
f2_256[38:218, 33:223] = np.copy(f2)

'''
Event stuff goes here
'''
print("Loading event data")
event_dir = "/home/srutarshi/Desktop/nDistortion_mult/Option2_fin_FasterRCNN-MultiClassKP/BMVC_2020/GEF_dataset/"+folder_p+"/events_clip"
ev_filenames = []
for root, dirs, files in os.walk(event_dir):
    for f in files:
        ev_filenames.append(root+"/"+f)

#print('ev_filenames', ev_filenames)
ev_filenames_sorted = natsorted(ev_filenames)
#print('sorted_filenames', ev_filenames_sorted)

print("[INFO] loading model...")

# end of line 41 in MATLAB code
print('Creating qt-related global variables...')
o = qt_orient(Nl)
a = qt_segment(Nl)
[f,t] = qt_sets(Nl)
sq = qt_squares_bboxes(Nl,o)
print('sq', len(sq), np.shape(sq[0]), np.shape(sq[1]), np.shape(sq[2]), np.shape(sq[3]), np.shape(sq[4]))
print('sq', np.shape(sq[5]), np.shape(sq[6]), np.shape(sq[7]), np.shape(sq[8]))
#print('sq_full', sq)

# end of line 55 in MATLAB code
f_rec = np.zeros((nf,new_rows,new_cols), dtype='uint8')
f_rec[0,:,:] = f2_256.copy()

rate = np.zeros((nf-1,1))
flow = np.zeros((nf-1,3,256,256))
qtsq = np.zeros((nf-1,3,256,256)) #save quadtree S & Q for video
segment = np.ones((nf-1,256,256))
segm_track = np.ones((nf-1,256,256))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rootdir_wp = os.path.join(rootd_det, rootdir_d_fi+'.mp4')
out = cv2.VideoWriter(rootdir_wp, fourcc, 30.0, (int(new_rows+0.5), int(new_cols+0.5)))

outfile_tracked = os.path.join(rootd_det, rootdir_d_fi+'_tracked.txt')
outfile_dets = os.path.join(rootd_det, rootdir_d_fi+'_detections.txt')
outfile_comp = os.path.join(rootd_det, rootdir_d_fi+'_compressed.txt')

fil = open(outfile_tracked, "a+")
fil1 = open(outfile_dets, "a+")
fil2 = open(outfile_comp, "a+")
k1 = 2

print('\n Starting simulation...')
for k in range(2, nf):
#for k in range(2, 4):
    event_file_c = ev_filenames_sorted[k-1]
    f2 = f_rec[k-2,:,:]

    # obtain new frame on Chip
    f3 = cv2.imread(int_filenames_sorted[k-1], 0)
    f3 = cv2.resize(f3, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    f3_256 = np.zeros(shape=(256, 256), dtype=np.uint8)
    f3_256[38:218, 33:223] = np.copy(f3)
    f3 = np.copy(f3_256)
    print('event_file_c', event_file_c, int_filenames_sorted[k-1])
    
    segm_track_p = segm_track[k-1,:,:]
    area_max = 256 * 256
    area = area_max

    # Run Viterbi on Host (if f3 is used - Viterbi is run on chip. If f3_rec is used - Viterbi is run on host)
    priorities = [0,2] #this is not used in MATLAB code (useless)
    
    lambda_range = [lambda0, lambda1]
    print('lambda_range', lambda_range)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        [ret0, ret1] = executor.map(qt_viterbi_segm_init, lambda_range)
    
    tree0 = ret0[0]; gxf_min0 = ret0[1]; total_rate0 = ret0[2]
    tree1 = ret1[0]; gxf_min1 = ret1[1]; total_rate1 = ret1[2]

    distort0 = gxf_min0 - (lambda0*total_rate0)
    print('lambda0',lambda0)
    print('lambda1',lambda1)
    distort1 = gxf_min1 - (lambda1*total_rate1)
    R0 = total_rate0
    D0 = distort0
    R1 = total_rate1
    D1 = distort1
    
    lambda_new = next_lambda_bezier(Rm, R0, D0, lambda0, R1, D1, lambda1)
    [tree_new, gxf_min_new, total_rate_new] = qt_viterbi_segm(f2,f3,N,n0,o,a,f,t,segm_track_p,priorities,lambda_new, area)
    R_new = total_rate_new
    D_new = gxf_min_new - (lambda_new * total_rate_new)
    print('lambda_new',lambda_new)

    while ((abs(Rm-R_new)/Rm) > 0.05):
        if (R_new > Rm):
            R0 = R_new
            D0 = D_new
            lambda0 = lambda_new
        else:
            R1 = R_new
            D1 = D_new
            lambda1 = lambda_new

        lambda_new = next_lambda_bezier(Rm, R0, D0, lambda0, R1, D1, lambda1)
        print('lambda_new',lambda_new)
        [tree_new, gxf_min_new, total_rate_new] = qt_viterbi_segm(f2,f3,N,n0,o,a,f,t,segm_track_p,priorities,lambda_new, area)
        R_new = total_rate_new
        D_new = gxf_min_new - (lambda_new * total_rate_new)

    print('lambda_final', lambda_new)
    lambda0 = lambda_new/150
    lambda1 = lambda_new*150
    tree = tree_new
    total_rate = total_rate_new
    
    a1 = (total_rate/Max_Rate)*100
    
    r_4 = 1; Nbins_T = 16; r_2 = 0; r_1 = 0
    s_a = 0; comp2 = 0; event_3D_last = np.zeros(shape = (256, 256, Nbins_T)); event_3D_last_sa = np.zeros(shape=(1,1))
    event_3D_last_sa = np.zeros(shape=(1,1))

    event_3D_pos, event_3D_neg, TOTAL_EVENTS_ORG, ppp, total_events, TOT_COMPR_BITS, BIT_STD1, PSNR, SSIM, T_error_ret = qt_events(tree, sq, event_file_c,\
    r_4, s_a, comp2, event_3D_last_sa, r_1, r_2, k-1, root_ev_path)     
    

    fil2.write(str(k-1)+','+str(TOTAL_EVENTS_ORG)+','+str(total_events)+','+str(TOT_COMPR_BITS)+','+str(BIT_STD1)+','+str(lambda_new)+','+str(PSNR)+\
    ','+str(SSIM)+','+str(T_error_ret)+','+str(np.count_nonzero(event_3D_pos))+','+str(np.count_nonzero(event_3D_neg)))
    fil2.write("\n")

    # Chip receives tree (S,Q) and performs reconstruction
    [tree_rec,f3_rec,e_rec] = qt_reconstruct_b(tree,sq,f2,f3)
    dist_rec = np.sum(abs(f3 - f3_rec))

    # Chip receives the events compressed and performs reconstruction
    #event_rec = qt_event_reconstruct(chip_host_events, s_a_final, Nbins_T, tree, sq, event_3D_last)

    f3_corr = f3_rec
    '''aa11 = (rate[k-1]/Max_Rate)*100'''
    #print('rate after correction: ',rate[k-3],aa11)

    # send corrected state and data back to host
    f_rec[k-1,:,:] = np.copy(f3_corr)
	
    # OpenCV does'nt save 2 dim frames - it has to be converted to 3 dim.
    f3_save = cv2.cvtColor(f3_corr,cv2.COLOR_GRAY2RGB)
    er_frame = cv2.resize(f3_save, (512, 512), interpolation=cv2.INTER_AREA)
    er_frame = f3_save

    out.write(er_frame)
    #event_3D_tr_pr = event_3D_tr

out.release()
#vs.release()

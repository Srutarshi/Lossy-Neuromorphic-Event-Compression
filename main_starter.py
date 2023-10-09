from __future__ import division
import numpy as np
import math as ma
import argparse
import time
import copy
import subprocess
import os


folder_dir = ["Indoor3"]
folder_len = len(folder_dir)
for ii in range(folder_len):
   aa = "python main.py " + folder_dir[ii]
   print('aa', aa)
   subprocess.call(aa, shell=True)

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sc
import math as mt
import scipy.ndimage as nd
from scipy.interpolate import lagrange
from scipy.signal import savgol_filter
import subprocess
import time

import random


#from codecs import decode
#import zstd as zstd

from mgard import MGARD
from mgard_compression import * #Compression pipeline, bitplans, zstd, huffman
from tools import * #Plotting functions, metrics, dataset generation
from mgard_adaptive import *


def cutting_gradient(votes,min_shape,pick=0):
	max_g = 0
	d=0
	x=0
	c=False
	for i in range(votes.ndim):
		if votes.shape[i] > min_shape[i]:
			lo = min_shape[i]
			hi = votes.shape[i]-min_shape[i]

			agg = np.count_nonzero(votes == pick, axis=1)
			#agg = savgol_filter(agg, min(51,agg.size), 3) # window size 51, polynomial order 3

			grad = abs(np.gradient(agg))
			m =np.max(  grad[lo:hi]  )
			if (m>max_g) or (m == max_g and votes.shape[d] < votes.shape[i]):
				max_g = m
				d = i
				x = lo+argmax(grad[lo:hi])
				c = True
	l = [0]*votes.ndim
	l[d]=x
	return c,l


def compare(f1,f2):
	with open(f1,"rb") as f:
		u1=np.fromfile(f,dtype="f")
	with open(f2,"rb") as f:
		u2=np.fromfile(f,dtype="f")
	uf = u2.flatten()
	u0 = u1.flatten()
        	
	e_1 = np.linalg.norm(abs(u0-uf),ord=np.inf) / np.linalg.norm(u0,ord=np.inf)
	e_a = np.linalg.norm(abs(u0-uf),ord=np.inf)
	e_2 = np.linalg.norm(abs(u0-uf),ord=2) / np.linalg.norm(u0,ord=2)
	with open("results.out","a") as f:
		f.write(f"Error Linf (rel):{e_1}\t(abs):{e_a}\tL2 (rel):{e_2}\n")

def fun_entropy(u):
	return -1 * entropy(u)


def example():
	# Load data (2D)

	u=np.load("data.npy")

	orders_list= [  [0]*u.ndim, [1]*u.ndim, [2]*u.ndim ]
	
	framework = MGARD_adaptive(thr=0.8,min_shape=[20]*u.ndim, fun_cut=cutting_gradient,fun_vote=fun_entropy,cell_size=1,orders_list=orders_list)
	
	
	coord_list,shape_list,vote_list = framework.decompose_blocks(u)
	u=adaptive_decompose(u,coord_list,shape_list,vote_list)
	compress_zstd(u)
	u=adaptive_recompose(u,coord_list,shape_list,vote_list)

	

def main(argv):
	example()
	


if __name__ == '__main__':
  main(sys.argv)

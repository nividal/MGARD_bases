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

	for i in votes.ndim:
		if votes,shape[i] > min_shape[i]:
			lo = min_shape[i]
			hi = votes.shape[i]-min_shape[i]

			agg = np.count_nonzero(votes == pick, axis=1)
			#agg = savgol_filter(agg, min(51,agg.size), 3) # window size 51, polynomial order 3

			grad = abs(np.gradient(agg))
			m =np.max(  grad[lo:hi]  )
			if (m>max_g) or (m == max_g and votes.shape[d] < votes.shape[i]):
				max_g = m
				d = i
				x = lo+argmax(grad[[lo:hi]])
	l = [0]*votes.ndim
	l[d]=x
	return l


def fun_maj(residuals):




def reconstruct_from_imported_coefs(u,mgard_grids,coord,coefs="coefs_decompressed.csv"):


	## Import coefs
	uc = np.zeros(u.shape)
	with open(coefs,"r") as f:
		for e,v in np.ndenumerate(uc):
			uc[e]=float(f.readline())

	for i in range(len(mgard_grids)):
		mg = mgard_grids[i]
		if u.ndim == 3:
			x,y,z = coord[i]
			sx,sy,sz = mg.original_shape
			mgard_grids[i].set_u(uc[x:x+sx,y:y+sy,z:z+sz].copy())
		if u.ndim == 2:
			x,y,z = coord[i]
			sx,sy = mg.original_shape
			mgard_grids[i].set_u(uc[x:x+sx,y:y+sy].copy())
		if u.ndim == 1:
			x,y,z = coord[i]
			sx = mg.original_shape
			mgard_grids[i].set_u(uc[x:x+sx].copy())

	#Reconstruct
	uf=np.zeros(u.shape)
	for i in range(len(mgard_grids)):
		mg = mgard_grids[i]
		mg.recompose_full()

		if u.ndim == 3:
			x,y,z = coord[i]
			sx,sy,sz = mg.original_shape
			uf[x:x+sx,y:y+sy,z:z+sz] = mg.get_u().copy()
		if u.ndim == 2:
			x,y,z = coord[i]
			sx,sy = mg.original_shape
			uf[x:x+sx,y:y+sy] = mg.get_u().copy()
		if u.ndim == 1:
			x,y,z = coord[i]
			sx = mg.original_shape
			uf[x:x+sx] = mg.get_u().copy()


	uf = uf.flatten()
	u0 = u.flatten()
	#Metrics

	e_1 = np.linalg.norm(abs(u0-uf),ord=np.inf) / np.linalg.norm(u0,ord=np.inf)
	e_a = np.linalg.norm(abs(u0-uf),ord=np.inf) 
	e_2 = np.linalg.norm(abs(u0-uf),ord=2) / np.linalg.norm(u0,ord=2)
	with open("results.out","a") as f:
		f.write(f"Error Linf (rel):{e_1}\t(abs):{e_a}\tL2 (rel):{e_2}\n")


	#Print error


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
	return -1 * entropy_2(u)


def example():
	# Load data (2D)

	u=np.load("data.npy")
	
	framework = MGARD_adaptive(signature=0.8,min_shape:[20]*u.ndim, fun_cut=cutting_gradient,fun_vote=fun_entropy,fun_sign=fun_maj,cell_size:1,fun_compression)
	
	#Grid
	coord_list,shape_list,vote_list = framework.decompose_block(u)
	mglist=framework.mgard_list(u,coord_list,shape_list,vote_list)
	u=adaptive_decompose(u,coord_list,shape_list)
	compress_zstd(u)
	u=adaptive_recompose(u,coord_list,vote_list):

	

def main(argv):
	pipeline(argv)
	


if __name__ == '__main__':
  main(sys.argv)

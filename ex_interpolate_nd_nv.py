import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import math as mt
import scipy.ndimage as nd

from codecs import decode
import struct
import zstd as zstd

from mgard import MGARD



#######
# bitplan
#######


def bin_to_float(b):
		""" Convert binary string to a float. """
		bf = int_to_bytes(int(b, 2), 8)
		return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  
	return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value): 
	""" Convert float to 64-bit binary string. """
	[d] = struct.unpack(">Q", struct.pack(">d", value))
	return '{:064b}'.format(d)


def bitplan_mask(u,mask):
		## Apply bitmask to all elements of u
		for e,v in np.ndenumerate(u):
				u[e] = bin_to_float(bin(int(float_to_bin(v),2) & mask))
		return u






##############
# Various metrics
##############

def compute_sparseness(u):
		ent=compute_entropy(u)
		gini=compute_gini(u)
		hoyer=compute_hoyer(u)
		tanh05=compute_tanh(u,1,1/2)
		tanh1=compute_tanh(u,1,1)
		tanh2=compute_tanh(u,1,2)

		l10=compute_le(u, np.max(u)/10)
		l100=compute_le(u, np.max(u)/100)
		l1l2=compute_l1l2(u)
		k4=compute_k4(u)
		slog=compute_slog(u)
		total_var=compute_total_var(u)

		## And more
		metrics=[ent,gini,hoyer,tanh05,tanh1,tanh2,l10,l100,l1l2,k4,slog,total_var]
		
		grd=compute_gradient(u)
		for g in grd:
				metrics.append(g)

		lp=compute_laplacian(u)
		for l in lp:
				metrics.append(l)
		return metrics

def compute_entropy(u,bins=500):
		hist, _ = np.histogram(u, bins=bins)
		prob_dist = hist / hist.sum()
		return sc.entropy(prob_dist, base=2)

def compute_gradient(u):
		res=[]
		grd=np.gradient(u)

		res.append(np.sum(grd))
		res.append(np.max(grd))
		res.append(np.average(grd))
		res.append(np.std(grd))
		return res

def compute_laplacian(u):
		lp=nd.laplace(u)
		#np.linalg.norm(lp,1),np.linalg.norm(lp,2),
		return [np.sum(lp),np.max(lp),np.average(lp),np.std(lp)]


def compute_gini(u):
		v=u.flatten()
		np.sort(v)
		N=len(v)
		s=0
		l1=np.sum(v)
		for k in range(len(v)):
				s+= v[k]/ l1 * (N-(k+1) + 1/2 )/N

		return 1-2*s


def compute_hoyer(u):
		s=0
		s2=0
		for i,v in np.ndenumerate(u):
				s+=v
				s2+= v**2
		return mt.sqrt(u.size) * mt.sqrt(s2) / s * (1 / (mt.sqrt(u.size) -1 ) )

#a=1, b = 0.5, 1, 2
def compute_tanh(u,a,b):
		s=0
		for i,v in np.ndenumerate(u):
				#print(a,v,b)
				s += np.tanh(pow((a*v),b))
		return -1*s

# 10-cil of the maximum
def compute_le(u,epsilon):
		n=0
		for i,v in np.ndenumerate(u):
				if v>epsilon:
						n+=1
		return n


def compute_l1l2(u):
		s=0
		s2=0
		for i,v in np.ndenumerate(u):
				s+=v
				s2+= v**2
		return mt.sqrt(s2)/s

def compute_k4(u):
		s4=0
		s2=0
		for i,v in np.ndenumerate(u):
				s4+=v**4
				s2+= v**2
		return mt.sqrt(s4)/(s2**2)



def compute_slog(u):
		slog=0
		for i,v in np.ndenumerate(u):
				slog+= mt.log(1+v**2)
		return -1 * slog


def compute_total_var(u):
		return np.sum(np.abs(np.diff(u)))



#####
# Compressions
#####

def compress_zstd(u, file='test.npz'):
	data=zstd.ZSTD_compress(u.tobytes())
	np.save(file,data)
	return len(data)


def save_by_bitplan(u,compression = 'default', file='test'):
	sizes=[]
	for i in range(64):
		mask = 2 ** i
		plane = bitplan_mask(u,mask)

		if compression == 'default':
			np.savez_compressed(plane.tobytes(),f'{file}_{i}.npz')

		if compression == 'zstd':
			sizes.append(compress_zstd(u,f'{file}_{i}.npz'))
		
		'''
		if compression == 'huffman':
			sizes.append(compress_huffman(u,f'{file}_{i}.npz'))
		'''
		return sizes


########
# Plotting function
########

def plot_u(u,path,name):
	uplot=[v for i,v in np.ndenumerate(u)]
	plt.figure()
	plt.plot(uplot)
	plt.gca().set_box_aspect(1)
	spath=Path("path")
	spath.mkdir(parents=True, exist_ok=True)
	plt.savefig(f"{path}/{name}", dpi=100, bbox_inches='tight')
	plt.close()
	return 0

########
### exp scripts
########

def exp_bitplan_compression(u, grid, name):
		print("exp_bitplan")


		plot_u(u, "Img/bitplan/","f{name}_original.pdf")


		metrics=compute_sparseness(u)

		ndims=u.ndim

		err_inf = {}
		uerr={}
		size_arr={}
		u0=u.flatten()

		o1=1
		dim=3

		########################
		for order in [0,1,2]:

				
				mg = MGARD(grid, u, order=[o1]*dim, order2=[order]*dim)
				########################


				if ndims==1:
						xm=len(mg.u_mg)
				if ndims==2:
						xm,ym=mg.u_mg.shape
				if ndims==3:
						xm,ym,zm=mg.u_mg.shape
				err_inf[order] = []
				size_arr[order]=[]


				mc=mg.u_mg.flatten()
				ind = np.argsort(abs(mc))
				
				mg.decompose_full()
				#float bits: 1 (sign), 11 (exposant) 52 significative



				plot_u(u,"Img/bitplan/",f'{name}_coeff{mg.order}_.pdf')



				mask= np.iinfo(np.uint64).max
				for i in range(-1,52):

						if (i>-1):
								mask = (mask - (2 ** i))
								mg.u_mg = bitplan_mask(mg.u_mg,mask)
						u_mg = mg.u_mg.copy()

						#np.savez_compressed('test.npz', *mg.u_mg)
						#size=os.stat('test.npz').st_size
						
						size=np.sum(save_by_bitplan(u,'zstd'))

						mg.recompose_full()


						plot_u(mg.u_mg,"Img/bitplan/",f"{name}_reconstructed_order{mg.order}_plan{i}.pdf")

						uc=mg.u_mg.flatten()
						err_inf[order].append(np.linalg.norm(abs(u0-uc),ord=np.inf) / np.linalg.norm(u0,ord=np.inf))
						size_arr[order].append(size)
						uerr[order]=np.zeros(u.shape)
						
						for x,v in np.ndenumerate(u):
								uerr[order][x] = abs(mg.u_mg[x]-v)
								
						mg.u_mg = u_mg         


		plt.figure()
		plt.semilogy(err_inf[0][-1::-1], label='order 0')
		plt.semilogy(err_inf[1][-1::-1], label='order 1')
		plt.semilogy(err_inf[2][-1::-1], label='order 2')
		plt.title('Truncation error')
		plt.legend()
		plt.xlabel('bitplan', fontsize='xx-large')
		plt.ylabel(r'$\|u-\tilde{u}\|_{\infty}/\|u\|_{\infty}$', fontsize='xx-large')
		plt.gca().set_box_aspect(1)
		plt.savefig("Img/{0}_bitplan.pdf".format(name), dpi=100, bbox_inches='tight')
		plt.close()

		plt.figure()
		plt.semilogy(size_arr[0][-1::-1], label='order 0')
		plt.semilogy(size_arr[1][-1::-1], label='order 1')
		plt.semilogy(size_arr[2][-1::-1], label='order 2')
		plt.title('Size of the array')
		plt.legend()
		plt.xlabel('bitplan', fontsize='xx-large')
		plt.ylabel('Size', fontsize='xx-large')
		plt.gca().set_box_aspect(1)
		plt.savefig("Img/{0}_size_bitplan.pdf".format(name), dpi=100, bbox_inches='tight')
		plt.close()
								

		diff0= np.sum(np.subtract(err_inf[0],err_inf[1]))
		diff2 = np.sum(np.subtract(err_inf[2],err_inf[1]))
		
		e0 = np.sum(err_inf[0])
		e1 = np.sum(err_inf[1])
		e2 = np.sum(err_inf[2])

		#metrics and norm
		with open('res_bitplan.txt','a') as f:
				f.write(f'{name} {diff0} {diff2} {e0} {e1} {e2}')
				for metric in metrics:
						f.write(f' {metric}')
				f.write('\n')

		return 0




def main(argv):
	## Main, run selected exp scripts, on input files
	for filename in os.listdir("100x500x500/"):
			print(filename)
			inputfile="100x500x500/"+filename
			name=filename.split('.')[0]
			if filename.split('.')[0] == 'log10':
					name = name+'.log10'
			u=np.fromfile(inputfile,dtype='<f').astype('f')

			
			#Very small subset, found a bug in non-square projection
			Nx = 33
			Ny = 33
			Nz = 33

			u.resize(Nx,Ny,Nz)


			grid = [np.linspace(0,1,Nx), np.linspace(0,1,Ny), np.linspace(0,1,Nz)]
			ind  = [np.arange(0,Nx), np.arange(0,Ny), np.arange(0,Nz) ]
			ind0 = [np.arange(0,Nx,2), np.arange(0,Ny,2), np.arange(0,Nz,2) ]
			dind = [np.arange(1,Nx,2), np.arange(1,Ny,2), np.arange(1,Nz,2) ]
			
			#exp_reconstruct(u,grid,name)
			#exp_compress_level(u,grid,filename)
			exp_bitplan_compression(u,grid,name)

	###




if __name__ == '__main__':
  main(sys.argv)


from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sc
import math as mt
import scipy.ndimage as nd
from scipy.interpolate import lagrange
from memory_profiler import profile
import gc
from entropy_estimators import continuous
import pandas as pd
import random



#################################################################################
# Various metrics
#################################################################################

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

		metrics.append(variogram(u,(0,1)))
		metrics.append(variogram(u,(1,0)))

		return metrics

def compute_entropy(u,bins=1000,span=0,file=None):
		if (span):
				bins=int((np.max(u)-np.min(u))/span)
				if bins==0:
					bins=1000
		hist, hbins = np.histogram(u, bins=bins,density=True)
		
		if (file):
			plt.figure()
			plt.hist(u, bins = hbins)
			plt.yscale("log")
			plt.savefig(file, dpi=100, bbox_inches='tight')
			plt.close()
		
		#prob_dist = hist / hist.sum()
		return sc.entropy(hist, base=2)

def entropy(u,rule='Sturge'):
	Dx=1000
	#https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information
	if rule == 'Sturge': #for large n
		Dx= mt.ceil(1+np.log2(u.size))
	if rule == 'Scott': #normalilty assumption
		if np.std(u)>0:
			Dx = mt.ceil((np.max(u)-np.min(u))/ ( 3.5 * np.std(u) * (u.size ** (-1/3))    ) )				
	if rule =='Freedman-Diaconis':
		q75, q25 = np.percentile(u, [75 ,25])
		iqr = q75 - q25
		if iqr>0:
			Dx = mt.ceil( (np.max(u)-np.min(u))/(2*iqr*(u.size ** (-1/3)))    )
	return compute_entropy(u,bins=Dx)
				


def compute_kl(u0,u1,bins=1000):
	hrange = min(np.min(u0),np.min(u1)),max(np.max(u0),np.max(u1))
	hist0, _ = np.histogram(u0,bins=bins,density=True,range=hrange)
	#prob0 = hist0 / hist0.sum()
	hist1, _ = np.histogram(u1,bins=bins,density=True,range=hrange)
	#prob1 = hist1 / hist1.sum()
	return sc.entropy(pk=hist1,qk=hist0)

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

###################################################################
# Entropy of coefficients
#####################################################################

def float_entropy(planes):
	mantissa=np.zeros(planes[0].shape,dtype=np.uint64)
	exponent=np.zeros(planes[0].shape,dtype=np.uint64)
	for i in range(52):
		mantissa += planes[i]*(2**i)

	for i in range(12):
		exponent+= planes[52+i]*2**i

	reconstructed_float=mantissa+exponent*(2**52)

	return compute_entropy(mantissa),compute_entropy(exponent),compute_entropy(	binarr_to_float(reconstructed_float),span=0.01)


def variogram(data,h):

	if data.ndim == 2:
		var = np.sum( (data[0:-1-h[0],0:-1-h[1]] - data[h[0]:-1,h[1]:-1]     )**2)
		return var / data[0:-1-h[0],0:-1-h[1]].size
	if data.ndim == 3:
		var = np.sum( (data[0:-1-h[0],0:-1-h[1],0:-1-h[2]] - data[h[0]:-1,h[1]:-1,h[2],-1]     )**2)
		return var / data[0:-1-h[0],0:-1-h[1],0:-1-h[2]].size

###################################################################
# Merge coefficients
#####################################################################

def merge(u,v,direction='right'):
	if u.ndim==2:
		xu,yu = u.shape
		xv,yv = v.shape

		if direction=='right':
			w=np.zeros((xu+xv,yu))
			w[0:xu,:] = u
			w[xu:,:] = v

		if direction=='below':
			w=np.zeros((xu,yu+yv))
			w[:,0:yu]=u
			w[:,yu:]=v
		return w.copy()
	if u.ndim==1:
		w=np.zeros((u.size+v.size))
		w[0:u.size]=u
		w[u.size:u.size+v.size]=v
		return w.copy()


########
# Plotting functions
########

#@profile
def plot_u(u,path,name,scale="default"):
	uplot=[v for i,v in np.ndenumerate(u)]
	plt.figure()
	if scale == "log":
		plt.yscale("log")
	plt.plot(uplot)
	plt.gca().set_box_aspect(1)
	spath=Path(path)
	spath.mkdir(parents=True, exist_ok=True)
	plt.savefig(f"{path}/{name}", dpi=100, bbox_inches='tight')
	plt.clf()
	plt.close()
	gc.collect()
	return 0


def plot_2D(u,path,name):
	plt.figure()
	plt.imshow(u)
	plt.gca().set_box_aspect(1)
	spath=Path(path)
	spath.mkdir(parents=True, exist_ok=True)
	plt.savefig(f"{path}/{name}", dpi=1000, bbox_inches='tight')
	plt.clf()
	plt.close()
	gc.collect()
	return 0


def plot_relative(grp,name,xlabel='',ylabel='',scale='',title='',labels=[]):
	#if grplabel==[]:
	#	grplabel=[""]*len(grp)
	plt.figure()
	fig, ax = plt.subplots()
	for i in range(len(grp)):
		x,y=grp[i]
		if labels != []:
			ax.scatter(x, y,label=f'{labels[i]}')
		else:
			ax.scatter(x, y,label=f'{i}')

	ax.legend()
	ax.grid(True)
	plt.gca().set_box_aspect(1)
	plt.xlabel(xlabel, fontsize='xx-large')
	plt.ylabel(ylabel, fontsize='xx-large')
	plt.title(title)
	if scale == 'log':
		plt.xscale("log")
		plt.yscale("log")
	plt.savefig(name,dpi=100, bbox_inches='tight')
	plt.close()
	return 0




###################################################################
## Dataset generation / Loading files
###################################################################




def gen_func(l,size=2**12+1,verbose=False):
	grid=np.linspace(-4*mt.pi,4*mt.pi,size)
	u = np.zeros((size))
	for i in range(len(l)):
		u[ int(i * size / len(l)) : int((i+1) * size/len(l))  ]= np.polynomial.chebyshev.chebval(np.cos(grid[ int(i * size / len(l)) : int((i+1) * size/len(l)) ]) , [1]*l[i])


	u *= mt.pi #to prevent bit patterns
	if (verbose):
 		plot_u(u,"Img","function.pdf")
	return u

def gen_func_2(l,steps,size=2**12+1,verbose=False,name='',ndim=1,seed=None):
	if name != '':
		name='/'+name
	#assume l.size divide steps
	random.seed()

	if ndim == 1:
		grid=np.linspace(0,1,size)
		u = np.zeros((size),dtype=np.float32)


		for i in range(steps):
			u[int(i*size/steps)] = random.uniform(0,10)
		#original_points=u[0:-1:int(size/steps)].copy()

		if (verbose):
			plot_u(u,"Img","function_basepoints.pdf")

		for i in range(len(l)):
			j = int(i * size / len(l))

			while( int(j+ (l[i] * size/steps)) <= size ) and (j+size/steps <= (i+1) * size/len(l) ):
				ind=[ int(j + k * size/steps) for k in range(l[i])]
				x=grid[ind]
				y=u[ind]
				poly=lagrange(x,y)
				u[ j : int (j+ size/steps)  ] = poly(grid[ j : int (j + size/steps)  ])
				j= int(j + size/steps)
		#u *= mt.pi #to prevent bit patterns

		if (verbose):
			plot_u(u,f"Img{name}","function.pdf")

	if ndim == 2:
		grid=[np.linspace(0,1,size),np.linspace(0,1,size)]
		u = np.zeros((size,size),dtype=np.float32)
		if seed != None:
			u_seed=np.load(seed)
			plot_2D(u_seed,"Img","fun_seed")
			u_seed = u_seed[::16,::16]
			seed_x,seed_y = u_seed.shape
			for i in range(steps):
				for j in range(steps):
					if int(i*size/steps/32) < seed_x and int(j*size/steps/32) < seed_y:
						u[int(i*size/steps),int(j*size/steps)] = u_seed[int(i*size/steps/32),int(j*size/steps/32)]
					else:
						u[int(i*size/steps),int(j*size/steps)] = rd.uniform(0,10)
		else:
			for i in range(steps):
				for j in range(steps):
					u[int(i*size/steps),int(j*size/steps)] = random.uniform(0,10)

		xl,yl = l.shape

		

		plot_2D(u,"Img","fun_2d_points")

		t_v = 0
		for gx in range(xl):
			for gy in range(yl):
				u_slice = u[ int(gx * size/xl) : int( (gx+1) * size/xl)+1 , int(gy * size/yl) : int( (gy+1) * size/yl)+1     ]
				#plot_2D(u_slice, "Img/fun",f"slice_{gx}x{gy}_1st")

				line = int(gx * size/xl)
				#while line%int(size/steps) != 0:
				#	line += 1
				while line < int((gx+1) * size/xl):
					#plot_u(u_slice[line,:],"Img/fun",f"line_{t_v}_{line}")
					i = (int(gy * size/yl))
					while  (int(i + (l[gx][gy]) * size/steps) <= size ) and  i+size/steps <= (gy+1)*size/yl :
						#while ( int(i + (l[gx][gy]) * size/steps) <=  int( (gy+1) * size/yl)):
						ind = [ int(i + k * size/steps)  for k in range(l[gx][gy]+1)]
						x=grid[1][ind]
						y=u[line,ind]

						poly=lagrange(x,y)
						u[ line, i : int(i + max((l[gx][gy]),1)*size/steps)   ] = poly(grid[1][i : int(i + max((l[gx][gy]),1)* size/steps)  ])
						#plot_u(u_slice[line,:],"Img/fun",f"line_{t_v}_{line}_o{l[gx,gy]}_interline")
						i += int(size/steps)

					#u[ line, i : -1   ] = poly(grid[1][i : -1  ])
					line+= int(size/steps)
		plot_2D(u,"Img/fun",f"fun_lines")

		#Columns
		for gx in range(xl):
			for gy in range(yl):
				#u_slice = u[ int(gx * size/xl) : int( (gx+1) * size/xl)+1 , int(gy * size/yl) : int( (gy+1) * size/yl)+1     ]
				column = int(gy * size/yl)
				while column < int((gy+1)*size/yl):
					i = int(gx * size/xl)
					while ( (int(i + (l[gx][gy]) * size/steps) <= size ) and  i + int(size/steps) <= (gx+1)*size/xl ):
						ind = [ int(i + k * size/steps)  for k in range(l[gx][gy]+1)]
						x=grid[0][ind]
						y=u[ind,column]

						poly=lagrange(x,y)
						u[ i : int(i + max((l[gx][gy]),1)* size/steps),column   ] = poly(grid[0][i : int(i + max((l[gx][gy]),1)*size/steps)  ])
						i += int(size/steps)

					column+= 1
		plot_2D(u,"Img/fun",f"fun_interpolate")

	## Plot with grid

	if ndim==2:
		plt.figure()
		fig, ax = plt.subplots()
		ax.imshow(u,alpha=0.80, interpolation='none',aspect='equal')
		ax.invert_yaxis()
		for gx in range(xl):
			for gy in range(yl):
				x =int(gx * size/xl)
				y = int(gy * size/yl)
				lx,ly = size/xl,size/yl
				xs = [y,y+ly,y+ly,y,y]
				ys = [x,x,x+lx,x+lx,x]
				ax.plot(xs, ys, color="red")

				ax.text(y+ly/2,x+lx/2,l[gx][gy])
				#ax.text(y/2+ly/2,x/2+lx/2,r)
		plt.savefig(f"Img/fun/fun_grid.pdf", bbox_inches='tight')



	u *= mt.pi
	return u


def gen_func_cheb(order,steps,size=2**12+1,name='',ndim=1,seed=None):
	if name != '':
		name='/'+name
	#assume l.size divide steps
	random.seed()

	if ndim == 1:
		grid=np.linspace(0,1,size)
		u = np.zeros((size))


		for i in range(steps):
			u[int(i*size/steps)] = random.uniform(0,10)
		#original_points=u[0:-1:int(size/steps)].copy()

		if (verbose):
			plot_u(u,"Img","function_basepoints.pdf")

			j=0

			while( int(j+ (order * size/steps)) <= size ) and (j+size/steps <= size ):
				ind=[ int(j + k * size/steps) for k in range(order)]
				x=grid[ind]
				y=u[ind]
				poly=lagrange(x,y)
				u[ j : int (j+ size/steps)  ] = poly(grid[ j : int (j + size/steps)  ])
				j= int(j + size/steps)
		u *= mt.pi #to prevent bit patterns

	if ndim == 2:
		grid=[np.linspace(0,1,size),np.linspace(0,1,size)]
		u = np.zeros((size,size))
		if seed != None:
			u_seed=np.load(seed)

			print("seed shape",u_seed.shape)

			seed_x,seed_y = u_seed.shape
			while seed_x > 2*size:
				u_seed = u_seed[::2,:]
				seed_x,seed_y = u_seed.shape
			while seed_y > 2*size:
				u_seed = u_seed[:,::2]
				seed_x,seed_y = u_seed.shape


			for i in range(steps):
				for j in range(steps):
					if int(i*size/steps/32) < seed_x and int(j*size/steps/32) < seed_y:
						u[int(i*size/steps),int(j*size/steps)] = u_seed[int(i*size/steps/32),int(j*size/steps/32)]
					else:
						u[int(i*size/steps),int(j*size/steps)] = random.uniform(0,10)
		else:
			for i in range(steps):
				for j in range(steps):
					u[int(i*size/steps),int(j*size/steps)] = random.uniform(0,10)

		u_slice = u
		line = 0

		while line < size:
			i = 0
			while  (int(i + (order) * size/steps) <= size ) and  i+size/steps <= size:
				ind = [ int(i + k * size/steps)  for k in range(order+1)]
				x=grid[1][ind]
				y=u[line,ind]

				poly=lagrange(x,y)
				u[ line, i : int(i + max(order,1)*size/steps)   ] = poly(grid[1][i : int(i + max(order* size/steps))  ])
				i += int(size/steps)

			line+= int(size/steps)

		#Columns

		#u_slice = u[ int(gx * size/xl) : int( (gx+1) * size/xl)+1 , int(gy * size/yl) : int( (gy+1) * size/yl)+1     ]
		column = 0
		while column < size:
			i = 0
			while ( (int(i + (order) * size/steps) <= size ) and  i + int(size/steps) <= size ):
				ind = [ int(i + k * size/steps)  for k in range(order+1)]
				x=grid[0][ind]
				y=u[ind,column]

				poly=lagrange(x,y)
				u[ i : int(i + max(order,1)* size/steps),column   ] = poly(grid[0][i : int(i + max(order,1)*size/steps)  ])
				i += int(size/steps)
			column+= 1
	u *= mt.pi
	return u


def func_from_file(file,shape=None,dim=None,dtype='<f'):


	uo=np.fromfile(file,dtype=dtype).astype(dtype)
	
	# Ugly drop of Nan/Inf values keeping data structure
	'''
	for e,v in np.ndenumerate(uo):
		if np.isinf(v):
			uo[e]=0
		if np.isnan(v):
			uo[e]=0
	'''

	if dim ==3:
		Ox,Oy,Oz=shape
		uo.resize(Ox,Oy,Oz)

		#Nx= 2 ** ( mt.ceil(mt.log2(Ox)) ) + 1 
		#Ny= 2 ** ( mt.ceil(mt.log2(Oy)) ) + 1 
		#Nz= 2 ** ( mt.ceil(mt.log2(Oz)) ) + 1
		u = uo.copy()
		#u=np.zeros(shape=(Nx,Ny,Nz))
		#u[0:min(Ox,Nx),0:min(Oy,Ny),0:min(Nz,Oz)]=uo[0:min(Nx,Ox),0:min(Oy,Ny),0:min(Nz,Oz)]

	if dim == 2:
		Ox,Oy=shape
		uo.resize(Ox,Oy)

		#Nx= 2 ** ( mt.ceil(mt.log2(Ox)) ) + 1 
		#Ny= 2 ** ( mt.ceil(mt.log2(Oy)) ) + 1 
		#u=np.zeros(shape=(Nx,Ny))
		#u[0:min(Ox,Nx),0:min(Oy,Ny)]=uo[0:min(Nx,Ox),0:min(Oy,Ny)]

		u = uo.copy()

	if dim == 1:
		Ox=shape
		uo.resize(Ox)

		#Nx= 2 ** ( mt.ceil(mt.log2(Ox)) ) + 1 
		#u=np.zeros(shape=(Nx))
		#u[0:min(Ox,Nx)]=uo[0:min(Nx,Ox)]

		u = uo.copy()

	return u




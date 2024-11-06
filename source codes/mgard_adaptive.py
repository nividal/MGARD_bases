from mgard import MGARD
from mgard_compression import * #Compression pipeline, bitplans, zstd, huffman
from tools import * #Plotting functions, metrics, dataset generation





def interpolation_error(u:np.ndarray,orders=list[int]):
	#orders is a list of interpolation orders sets (one for each dimension)
	grid = [np.linspace(0,1,u.shape[i]) for i in range(u.ndim)]

	mg=MGARD(grid,u.copy(),order=orders,order2=orders)
	#decompose 1st level and get coefficients
	_,_,dind,_=mg.decompose_level(1)

	if u.ndim == 1:
		coarse=mg.u_mg[dind[0]].copy()
	elif u.ndim == 2:
		coarse=mg.u_mg[dind[0][:,None],dind[1]].copy()
	elif u.ndim == 3:
		coarse=mg.u_mg[dind[0][:,None,None],dind[1][None,:,None],dind[2]].copy()
	elif u.ndim == 4:
		coarse=mg.u_mg[dind[0][:,None,None,None],dind[1][None,:,None,None],dind[2][None,None,:,None],dind[3]].copy()	
	return coarse

#Voting functions returning a vote and a signature from coarse coefficients

def get_grid(grid,coord:tuple,shape:tuple):
	#for some reasons
	coord=tuple(coord)
	shape=tuple(shape)

	sl = tuple( [slice(coord[i],coord[i] + shape[i],1) for  i in range(grid.ndim)] )
	return grid[sl]


def set_grid(grid,coord,shape,u):
	sl = tuple( [slice(coord[i],coord[i] + shape[i],1) for  i in range(grid.ndim)] )
	grid[sl] = u
	return grid 

def add_coord(coord,pos):
	for i in range(len(coord)):
		if pos[i]:
			coord[i]=pos[i]
	return coord

def split_shape( shape,coord,pos  ):
	s2 = shape
	for i in range(len(pos)):
		if pos[i]:
			s2= shape[i] - pos[i] + coord[i]
			shape[i] = pos[i]-coord[i]
	return shape,s2

def maj_score(vote,orders):
	sc = 0
	for e,v in np.ndenumerate(vote):
		sc += (v == tuple(orders))
	return sc/vote.size


class MGARD_adaptive(object):

	def __init__(self, thr:float,min_shape:list[int], fun_cut,fun_vote,cell_size:int,orders_list):
		#self.blocks=blocks
		self.thr=thr
		self.min_shape=min_shape

		self.fun_cut=fun_cut #return the axis coordinate where to cut ie (0,0,...,0,x,0,...,0)
		self.fun_vote=fun_vote #voting function, given residuals and orders provide signature

		self.cell_size=cell_size

		self.orders_list = orders_list


	##
	def iscutable(self,shape):
		b = 0
		for i in range(len(shape)):
			b |= (shape[i] > self.min_shape[i]) 
		return b

	##
	# Construct array of votes
	## 

	def compute_vote(self,u):
		shape = tuple([ int(u.shape[i]/(2*self.cell_size)) for i in range(u.ndim)  ])
		vote_grid = np.empty(shape,dtype=tuple)
		bv = np.empty(shape) 
		for orders in self.orders_list:
			c=interpolation_error(u,orders)
			for it in np.ndindex(shape):
				ind = tuple( [slice(  it[i]*self.cell_size ,  (it[i]+1)*self.cell_size   , 1   ) for i in range(u.ndim)] )
				v = np.linalg.norm(abs(c[ind]),ord=2)
				if v < bv[it]:
					bv[it] = v
					cf=c
					vote_grid[it] = tuple(orders)
		return vote_grid

	##
	# Map vote to a dataset using the voting function
	##
	def map_vote(self, u:np.ndarray  ):
		sign = -mt.inf
		vote = 0
		for orders in self.orders_list:
			c=interpolation_error(u,orders)
			s = self.fun_vote(c)
			if s>sign:
				sign = s
				vote = orders
				cf=c
		return vote,cf

	##
	# Decompose using Berger-Rigoustos algorithm
	##
	def decompose_blocks(self, u:np.ndarray):
		# Step 1: get 1st level decomposition

		#v,sign,_=self.map_vote(u)

		#grid_list = [u]
		vote_list=[ tuple([0] *u.ndim) ]
		sign_list=[-1]

		s2 = tuple([ int(u.shape[d]/2) for d in range(u.ndim)  ])
		residuals = np.empty(s2)

		coord_list=[ tuple([0]*u.ndim) ]
		shape_list= [ u.shape ]
		vote_grid = self.compute_vote(u)


		i=0
		while i<len(coord_list):
			grid=get_grid(u,coord_list[i],shape_list[i])
			v,res=self.map_vote(grid)

			cv = tuple([mt.ceil(coord_list[i][j]/2) for j in range(u.ndim)])
			sv = tuple([mt.ceil(shape_list[i][j]/2) for j in range(u.ndim)])

			sign=maj_score(get_grid(vote_grid,cv,sv),v)
			vote_list[i]=v
			#sign_list[i]=sign

			#Update residuals
			ind = tuple(  [ slice( cv[k],cv[k]+sv[k],1) for k in range(u.ndim)  ]  )
			residuals[ind] = res

			if self.iscutable(shape_list[i]): #has at least one dimension larger than the minimum shape
				if sign < self.thr: #threshold not reached
					pos=self.fun_cut(vote_grid,self.min_shape)

					if pos:

						#r1,r2=split_grid(res_list[i],pos)

						coord_list = coord_list + [  add_coord(coord_list[i],pos)   ]
						s1,s2 = split_shape( shape_list[i],coord_list[i],pos  )
						shape_list[i] = s1
						shape_list = shape_list + [ s2 ]

						vote_list = vote_list + [0] #updated later in the loop
						#sign_list = sign_list + [0]
						i -= 1 #cancel incr
			i +=1
		##
		# Merging step
		##
		i=0
		while i<len(coord_list):
			iterate_i = i+1
			j=0
			while j  < len(coord_list):
				iterate_j = j+1

				if i != j:
					if vote_list[i] == vote_list[j]:
						s1,s2=shape_list[i],shape_list[j]
						c1,c2=coord_list[i],coord_list[j]

						shared_edge=-1

						for ax in range(u.ndim):
							if s1[ax]==s2[ax] and c1[ax]==c2[ax]:
								pass
							elif c2[ax] == c1[ax]+s1[ax] and shared_edge==-1:
								shared_edge = ax
							else:
								shared_edge = -1
								break


						if shared_edge > -1:

							shape_list[i][shared_edge] += shape_list[j][shared_edge]

							del shape_list[j]
							del coord_list[j]
							del vote_list[j]
							iterate_i=0
							iterate_j=0
				j=iterate_j
			i=iterate_i


		return coord_list,shape_list,vote_list
	
	def mgard_list(u,coord_list,shape_list,vote_list):
		mglist = []
		for i in range(len(coord_list)):
			ui=get_grid(grid,coord[i],shape[i])
			grid = [np.linspace(0,1,shape[i][d]) for d in range(u.ndim) ] 
			mg=MGARD(grid,ui,order=[0]*dim,order2=[0]*dim)
			mglist.append(mg)

		return mglist

	def mgard_decompose_list(mglist,coord_list,shape_list): 
		for i in range(len(mglist)):
			mg.decompose_full()

	def adaptive_decompose(u,coord_list,shape_list,vote_list):
		for i in range(len(coord_list)):
			ui=get_grid(u,coord[i],shape[i])
			grid = [np.linspace(0,1,shape[i][d]) for d in range(u.ndim) ] 
			mg=MGARD(grid,ui,order=vote_list[i],order2=vote_list[i])
			mg.decompose_full()

			set_grid(u,coord[i],shape[i],mg.get_u)
		return u


	def adaptive_recompose(mglist,coord_lis,vote_list):
		for i in range(len(coord_list)):
			ui=get_grid(u,coord[i],shape[i])
			grid = [np.linspace(0,1,shape[i][d]) for d in range(u.ndim) ] 
			mg=MGARD(grid,ui,order=vote_list[i],order2=vote_list[i])
			mg.decompose_grid()
			mg.mgard_recompose_full()

			set_grid(u,coord[i],shape[i],mg.get_u)
		return u
		#reconstruct u

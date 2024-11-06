from typing import Sequence

import numpy as np
from itertools import product

import matplotlib.pyplot as plt
import math as mt


# def Gramm_matrix(grid, order=1):
# 	''' Gramm matrix for piecewise polynomial of given order on a given grid '''

# 	dgrid = np.diff(grid)
# 	if order==1:
# 		main_diag = np.concatenate((dgrid[0:1]/3,(dgrid[:-1]+dgrid[1:])/3,dgrid[-1:]/3))
# 		return np.diag(dgrid/6,-1) + np.diag(main_diag) + np.diag(dgrid/6,1)
# 	elif order==2:
# 		dx = dgrid[0]
# 		N = grid.size
# 		main_diag = 8/15*dx*np.ones((N,))
# 		main_diag[1::2] *= 2
# 		main_diag[0] /= 2
# 		main_diag[-1] /= 2
# 		sup_diag = 2/15*dx*np.ones((N-1,))
# 		supsup_diag = np.zeros((N-2,))
# 		supsup_diag[1::2] -= dx/15
# 		return np.diag(supsup_diag,-2) + np.diag(sup_diag,-1) + np.diag(main_diag) + np.diag(sup_diag,1) + np.diag(supsup_diag,2)





def element_Gramm_matrix(points: np.ndarray, order: int) -> np.ndarray:
	'''1d element fine-to-coarse Gramm matrix for piecewise polynomials of given order

		Inputs
		------
		  points: points on the fine element
		  order:  order of the polynomial basis
	'''

	# number of points in fine element
	Npf = len(points)

	if order==0:
		if Npf!=3:
                        
			raise ValueError("Mismatch between the number of nodes and the order of the element")
	elif Npf!=(2*order+1):
                raise ValueError("Mismatch between the number of nodes and the order of the element")

	# fine and coarse mesh sizes
	hf = np.diff(points)
	hc = np.diff(points[::2])

	if order==0:
		Ge = np.array([[hc[0], hf[1]]])
	else:
		Ge = np.zeros((order+1,2*order+1))
		if order==1:
			hc0, = hc
			hf0, hf1 = hf
			#
			Ge[0,0] = hc0/3.
			Ge[0,1] = (hc0+hf1)/6.
			Ge[0,2] = hc0/6.
			#
			Ge[1,0] = hc0/6.
			Ge[1,1] = (hc0+hf0)/6.
			Ge[1,2] = hc0/3.
		elif order==2:
			hc0, hc2 = hc
			hf0, hf1, hf2, hf3 = hf
			#
			Ge[0,0] = (hc0+hc2)*(6*hc0**2-3*hc0*hc2+hc2**2)/(30*hc0**2)
			Ge[0,1] = hc0**3*(3*hc0+5*hc2)/(60*(hc0+hc2)*hf0*hf1)
			Ge[0,2] = (hc0+hc2)**3*(2*hc2-3*hc0)/(60*hc0**2*hc2)
			Ge[0,3] = -hc2**5/(30*hc0*(hc0+hc2)*hf2*hf3)
			Ge[0,4] = -(hc0+hc2)*(3*hc0**2-4*hc0*hc2+3*hc2**2)/(60*hc0*hc2)
			#
			Ge[1,0] = Ge[0,2]
			Ge[1,1] = hc0**3*(2*hc0+5*hc2)/(60*hc2*hf0*hf1)
			Ge[1,2] = (hc0+hc2)**5/(30*hc0**2*hc2**2)
			Ge[1,3] = hc2**3*(5*hc0+2*hc2)/(60*hc0*hf2*hf3)
			Ge[1,4] = (hc0+hc2)**3*(2*hc0-3*hc2)/(60*hc0*hc2**2)
			#
			Ge[2,0] = Ge[0,4]
			Ge[2,1] = -hc0**5/(30*(hc0+hc2)*hc2*hf0*hf1)
			Ge[2,2] = Ge[1,4]
			Ge[2,3] = hc2**3*(5*hc0+3*hc2)/(60*(hc0+hc2)*hf2*hf3)
			Ge[2,4] = (hc0+hc2)*(hc0**2-3*hc0*hc2+6*hc2**2)/(30*hc2**2)
			# print(Ge[1,4],Ge[2,0],Ge[0,4])
			# print(Ge)
			# exit()
		# else:
			# x = np.linspace(0,1,Npf+1)
			# y = [lagrange(x,r)(x[i]) for i,r in enumerate(np.eye(Npf+1))]
			# print(y)
			# exit()
			# # for i_c in range(order+1):
			# # 	for i_f in range(2*order+1):
			# # 		lagrange(points[])
			# # 		Ge[i_c,i_f] =
	return Ge



def Gramm_matrix(grid: np.ndarray, order: int = 1) -> np.ndarray:
	'''1d fine-to-coarse Gramm matrix for piecewise polynomial of given order on a given grid

		Inputs
		------
		  grid:  1-d grid points on the fine grid
		  order: order of the polynomial basis
	'''

	# number of fine grid points
	Ngf = len(grid)

	if order==0:
		if (Ngf-1)%2!=0:
			raise ValueError("Mismatch between the number of nodes in a grid and the order of the element")

		# number of coarse elements
		Nec = (Ngf-1)//2

		# fine-to-coarse Gramm matrix
		Gcf = np.zeros((Nec+1,2*Nec+1))

		# loop through coarse elements
		for ec in range(Nec):
			Ge = element_Gramm_matrix(grid[2*ec:2*ec+3], order)
			Gcf[ec:ec+1,2*ec:2*ec+2] += Ge
		Gcf[-1,-1] = 1.0
	else:
		if (Ngf-1)%(2*order)!=0:
			raise ValueError("Mismatch between the number of nodes in a grid and the order of the element")

		# number of coarse elements
		Nec = (Ngf-1)//(2*order)

		# fine-to-coarse Gramm matrix
		Gcf = np.zeros((Nec*order+1,2*Nec*order+1))

		# loop through coarse elements
		for ec in range(Nec):
			# index of leftmost dof
			ei = order*ec
			Ge = element_Gramm_matrix(grid[2*ei:2*(ei+order)+1], order)
			Gcf[ei:ei+order+1,2*ei:2*(ei+order)+1] += Ge

	return Gcf


###############################################################################
###############################################################################


# def interpolation_matrix(grid, order):
# 	'''Interpolation matrix for piecewise polynomial of given order on a given grid '''
# 	pass


###############################################################################
###############################################################################


class MGARD(object):
	def __init__(self, grid: list[np.ndarray], u: np.ndarray, order: int = 1, order2 = None):

		self.original_shape = u.shape

		# Works with squares to avoid any boundary errors


	 #, interp='left'):
		self.grid   = grid
		self.u      = u.copy()
		self.grids = []
		self.u_mg   = u.copy()
		self.order  = order
		if order2 is None:
			self.order2 = order
		else:
			self.order2 = order2
		# self.interp = interp

		self.ndim = u.ndim

		d = 0
		for s in u.shape:
			if s != 2 ** ( mt.floor(mt.log2(s)) ) + 1:
				s = 2 ** ( mt.ceil(mt.log2(s)) ) + 1
			d = max(s,d)

		self.u_mg = np.zeros( tuple([d]*u.ndim) )
		sl = tuple([slice(0,u.shape[i]) for i in range(u.ndim)])
		self.u_mg[sl]=u.copy()

		self.u=self.u_mg.copy()
		self.grid = [ np.linspace(0,1,d) for i in range(u.ndim) ]


	def get_u(self):
		ind = tuple( [slice(0,self.original_shape[i]) for i in range(u.ndim) ])
		return self.u_mg[ind].copy()
	


	def interpolate_1d(self, grid, order, indc, dind, ind_f, ind_c):
		'''Interpolate values from coarse grid to surplus grid in-place

		Inputs
		------
		  grid:  1d grid nodes
		  order: 1d order
		  dind:	indices of the surplus nodes
		  indc:	indices of the fine    nodes
		  indc:	indices of the coarse  nodes
		'''

		if order==0:
			# loop through the 1d coarse constant elements along the given dimension
			for i in range(0,len(dind)):
				ind0 = np.ix_(*(ind_f+[[indc[i+0]]]+ind_c))
				indj = np.ix_(*(ind_f+[[dind[i+0]]]+ind_c))

				# interpolant
				self.u[indj] = self.u[ind0]


		elif order==1:
			# loop through the 1d coarse linear elements along the given dimension
			for i in range(0,min(len(dind),len(indc)-1),order):
				# coarse mesh step
				h = (grid[indc[i+1]] - grid[indc[i]])

				# 1d Lagrange basis functions
				l0 = -(grid[dind[i]] - grid[indc[i+1]]) / h
				l1 =  (grid[dind[i]] - grid[indc[i+0]]) / h

				######################################################
				#       this is the dimension to be interpolated
				#                     |     ||    |
				#                     |     \/    |
				ind0 = np.ix_(*(ind_f+[[indc[i+0]]]+ind_c))
				ind1 = np.ix_(*(ind_f+[[indc[i+1]]]+ind_c))
				indj = np.ix_(*(ind_f+[[dind[i+0]]]+ind_c))

				# interpolant
				self.u[indj] = self.u[ind0]*l0 + self.u[ind1]*l1

		elif order==2:

			# loop through 1d coarse quadratic elements along the given dimension
			for i in range(0,min(len(dind),len(indc)-2),order):

				# mesh steps

				h01 = (grid[indc[i+0]] - grid[indc[i+1]])
				h02 = (grid[indc[i+0]] - grid[indc[i+2]])
				#
				h12 = (grid[indc[i+1]] - grid[indc[i+2]])



				ind0 = np.ix_(*(ind_f+[[indc[i+0]]]+ind_c))
				ind1 = np.ix_(*(ind_f+[[indc[i+1]]]+ind_c))
				ind2 = np.ix_(*(ind_f+[[indc[i+2]]]+ind_c))

				# one point per interval of the element
				for j in range(order):
					# Lagrange basis functions
					l0 =  (grid[dind[i+j]] - grid[indc[i+1]]) * (grid[dind[i+j]] - grid[indc[i+2]]) / (h01 * h02)
					l1 = -(grid[dind[i+j]] - grid[indc[i+0]]) * (grid[dind[i+j]] - grid[indc[i+2]]) / (h01 * h12)
					l2 =  (grid[dind[i+j]] - grid[indc[i+0]]) * (grid[dind[i+j]] - grid[indc[i+1]]) / (h02 * h12)

					indj = np.ix_(*(ind_f+[[dind[i+j]]]+ind_c))

					# interpolant
					self.u[indj] = self.u[ind0]*l0 + self.u[ind1]*l1 + self.u[ind2]*l2
		return self.u


	def project_1d(self, grid, order, ind0, dind, ind_f, ind_c):
		'''Project function on a surplus grid to coarse grid

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		  ud:	values at the surplus nodes
		'''
		f = np.zeros(len(ind0))

		G = Gramm_matrix(grid[ind0], order)


		d0 = np.diff(grid[ind0])
		d1 = grid[ind0[1:]] - grid[dind]
		d2 = grid[dind]     - grid[ind0[:-1]]

		# contribution from dof to the left
		al = (2*d0 - d1) / 6
		# contribution from dof to the right
		ar = (2*d0 - d2) / 6


		# ind1 = np.ix_(*(ind_f+[[ind0[i+0]]]+ind_c))
		# ind2 = np.ix_(*(ind_f+[[ind0[i+1]]]+ind_c))
		# ind3 = np.ix_(*(ind_f+[[dind[i+0]]]+ind_c))

		ind4 = np.ix_(*(ind_f+[dind]+ind_c))

		ud = self.u[ind4]

		print(ud.shape)
		print(len(ind0))

		exit()

		# forcing term
		f[0]    =                     ar[0]  * ud[0]
		f[1:-1] = al[:-1] * ud[:-1] + ar[1:] * ud[1:]
		f[-1]   = al[-1]  * ud[-1]


		self.u[ind4] = np.linalg.solve(G,f[ind4])

		return np.linalg.solve(G,f)


	def interpolate_nd(self, indf, indc, dind):
		'''Interpolate values from coarse grid to surplus grid in-place

		Inputs
		------
		  indf: indices of the fine    nodes in each dimension
		  indc:	indices of the coarse  nodes in each dimension
		  dind:	indices of the surplus nodes in each dimension
		'''

		# loop through dimensions
		for d in range(self.ndim):
			# coarse and surplus indices along the given dimension
			indc_d = indc[d]
			dind_d = dind[d]

			# 1d grid along the given dimension
			grid_d = self.grid[d]

			# order along the given dimension
			order_d = self.order[d]

			# fine and coarse grid indices
			ind_f = [i0 for i0 in indf[:d]]
			ind_c = [i0 for i0 in indc[d+1:]]

			self.interpolate_1d(grid_d.copy(), order_d, indc_d.copy(), dind_d, ind_f, ind_c)
		return self.u


	def project_nd(self, indf, indc, dind):
		'''Project function defined on a surplus grid to coarse grid

		Inputs
		------
		  indf: indices of the fine    nodes in each dimension
		  indc:	indices of the coarse  nodes in each dimension
		  dind:	indices of the surplus nodes in each dimension
		'''

		u = self.u[np.ix_(*indf)].copy()

		# loop through dimensions
		for d in range(self.ndim):
			# coarse and surplus indices along the given dimension
			indc_d = indc[d]
			indf_d = indf[d]
			dind_d = dind[d]

			# 1d grid along the given dimension
			grid_d = self.grid[d]

			# order along the given dimension
			order_d = self.order[d]

			# fine and coarse grid indices
			ind_f = [i0 for i0 in indf[:d]]
			ind_c = [i0 for i0 in indc[d+1:]]

			G = Gramm_matrix(grid_d[indf_d], order_d)
			M = np.linalg.solve(G[:,::2],G[:,:])

			# # plt.spy(np.kron(G[:,::2],G[:,::2]), marker='.', markersize=5)
			# plt.spy(G[:,::2], marker='.', markersize=5)
			# plt.savefig('sparsity_2.png', dpi=100, bbox_inches='tight')
			# plt.show()
			# exit()

			uc = u.copy()
			ud = [uc.shape[d] for d in range(uc.ndim)]
			ud[d]=M.shape[1]
			uc.resize(ud)

			u = np.apply_along_axis(lambda x:M@x, d, uc)
		return u





	def interpolate(self, ind0, dind, u0):
		'''Interpolate values from coarse grid to surplus grid

		Inputs
		------
		  ind0:	indices of the coarse  nodes in each dimension
		  dind:	indices of the surplus nodes in each dimension
		  u0:	values at the coarse nodes
		'''

		# if len(ind0)!=self.ndim:
		# 	raise ValueError(f"list of indices ind0 must have indices for each out of {self.ndim} dimensions, got len(ind0) = {len(ind0)}")
		# if len(dind)!=self.ndim:
		# 	raise ValueError(f"list of indices dind must have indices for each out of {self.ndim} dimensions, got len(dind) = {len(dind)}")


		n_dind = len(dind)

		if self.order==0:
			return u0[:-1]
			# left dof (cadlag)
			# P[:,:-1] = np.eye(len(dind))
			# P[:,1:] = np.eye(len(dind))

			# if self.interp=='mid':
			# 	return 0.5*(u0[:-1]+u0[1:])
			# else:
			# 	return u0[:-1]
			# return u0[1:]
			# alpha = 0.3
			# return alpha*u0[:-1] + (1-alpha)*u0[1:]
		elif self.order==1:
			res = np.zeros(n_dind,)

			# loop through linear elements
			for i in range(n_dind):
				# mesh step
				h = (self.grid[ind0[i+1]] - self.grid[ind0[i]])

				# Lagrange basis functions
				l0 = -(self.grid[dind[i]] - self.grid[ind0[i+1]]) / h
				l1 =  (self.grid[dind[i]] - self.grid[ind0[i+0]]) / h

				# interpolant
				res[i] = u0[i] * l0 + u0[i+1] * l1
			return res

			# dgrid = np.diff(self.grid[ind0])
			# # interpolation matrix
			# P = np.zeros((len(dind),len(ind0)))
			# # left dof
			# P[:,:-1] = np.diag((self.grid[ind0][1:]-self.grid[dind])/dgrid)
			# # right dof
			# P += np.diag((self.grid[dind]-self.grid[ind0][:-1])/dgrid,1)[:-1,:]
			# return P @ u0
		elif self.order==2:
			res = np.zeros(n_dind,)

			# loop through quadratic elements
			for i in range(0,n_dind,self.order):
				# mesh steps
				h01 = (self.grid[ind0[i+0]] - self.grid[ind0[i+1]])
				h02 = (self.grid[ind0[i+0]] - self.grid[ind0[i+2]])
				#
				h12 = (self.grid[ind0[i+1]] - self.grid[ind0[i+2]])

				# one point per interval of the element
				for j in range(self.order):
					# Lagrange basis functions
					l0 =  (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) / (h01 * h02)
					l1 = -(self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) / (h01 * h12)
					l2 =  (self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) / (h02 * h12)

					# interpolant
					res[i+j] = u0[i]*l0 + u0[i+1]*l1 + u0[i+2]*l2
			return res
			# P = np.array([[3,6,-1],[-1,6,3]]) / 8
			# for i in range(0,len(dind),2):
			# 	res[i:i+2] = P @ u0[i:i+3]
			# return res
		elif self.order==3:
			res = np.zeros(n_dind,)

			# loop through cubic elements
			for i in range(0,n_dind,self.order):
				# mesh steps
				h01 = (self.grid[ind0[i+0]] - self.grid[ind0[i+1]])
				h02 = (self.grid[ind0[i+0]] - self.grid[ind0[i+2]])
				h03 = (self.grid[ind0[i+0]] - self.grid[ind0[i+3]])
				#
				h12 = (self.grid[ind0[i+1]] - self.grid[ind0[i+2]])
				h13 = (self.grid[ind0[i+1]] - self.grid[ind0[i+3]])
				#
				h23 = (self.grid[ind0[i+2]] - self.grid[ind0[i+3]])
				# one point per interval of the element
				for j in range(self.order):
					# Lagrange basis functions
					l0 =  (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) / (h01 * h02 * h03)
					l1 = -(self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) / (h01 * h12 * h13)
					l2 =  (self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) / (h02 * h12 * h23)
					l3 = -(self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) / (h03 * h13 * h23)

					# interpolant
					res[i+j] = u0[i]*l0 + u0[i+1]*l1 + u0[i+2]*l2 + u0[i+3]*l3
			return res
		elif self.order==4:
			res = np.zeros(n_dind,)

			# loop through cubic elements
			for i in range(0,n_dind,self.order):
				# mesh steps
				h01 = (self.grid[ind0[i+0]] - self.grid[ind0[i+1]])
				h02 = (self.grid[ind0[i+0]] - self.grid[ind0[i+2]])
				h03 = (self.grid[ind0[i+0]] - self.grid[ind0[i+3]])
				h04 = (self.grid[ind0[i+0]] - self.grid[ind0[i+4]])
				#
				h12 = (self.grid[ind0[i+1]] - self.grid[ind0[i+2]])
				h13 = (self.grid[ind0[i+1]] - self.grid[ind0[i+3]])
				h14 = (self.grid[ind0[i+1]] - self.grid[ind0[i+4]])
				#
				h23 = (self.grid[ind0[i+2]] - self.grid[ind0[i+3]])
				h24 = (self.grid[ind0[i+2]] - self.grid[ind0[i+4]])
				#
				h34 = (self.grid[ind0[i+3]] - self.grid[ind0[i+4]])
				# one point per interval of the element
				for j in range(self.order):
					# Lagrange basis functions
					l0 =  (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+4]]) / (h01 * h02 * h03 * h04)
					l1 = -(self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+4]]) / (h01 * h12 * h13 * h14)
					l2 =  (self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+4]]) / (h02 * h12 * h23 * h24)
					l3 = -(self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+4]]) / (h03 * h13 * h23 * h34)
					l4 =  (self.grid[dind[i+j]] - self.grid[ind0[i+0]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+1]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+2]]) * (self.grid[dind[i+j]] - self.grid[ind0[i+3]]) / (h04 * h14 * h24 * h34)

					# interpolant
					res[i+j] = u0[i]*l0 + u0[i+1]*l1 + u0[i+2]*l2 + u0[i+3]*l3 + u0[i+4]*l4
			return res


	def project(self, ind0, dind, ud):
		'''Project function on a surplus grid to coarse grid

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		  ud:	values at the surplus nodes
		'''
		f = np.zeros(len(ind0))
		if self.order==1:
			G = Gramm_matrix(self.grid[ind0], self.order)
			#
			d0 = np.diff(self.grid[ind0])
			d1 = self.grid[ind0[1:]] - self.grid[dind]
			d2 = self.grid[dind]     - self.grid[ind0[:-1]]
			# contribution from dof to the left
			al = (2*d0 - d1) / 6
			# contribution from dof to the right
			ar = (2*d0 - d2) / 6
			# forcing term
			f[0]    =                     ar[0]  * ud[0]
			f[1:-1] = al[:-1] * ud[:-1] + ar[1:] * ud[1:]
			f[-1]   = al[-1]  * ud[-1]
			return np.linalg.solve(G,f)
		elif self.order==2:
			G = Gramm_matrix(self.grid[ind0], self.order)
			#
			dx = np.diff(self.grid[ind0])[0]
			#
			# f[1] = 7/15*dx * (ud[0]+ud[1])
			# f[2] = -1/6*ud[0] - 4/15*dx*ud[1] - 4/15*dx*ud[2] - -1/6*ud[3]
			f[0]      =  4/15 * dx * ud[0] - 1/15 * dx * ud[1]
			f[1::2]   =  7/15 * dx * (ud[::2]+ud[1::2])
			f[2:-1:2] = dx/15 * ( -ud[:-3:2] + 4*ud[1:-2:2] + 4*ud[2:-1:2] - ud[3::2])
			f[-1]     =  -1/15 * dx * ud[-2] + 4/15 * dx * ud[-1]
			return np.linalg.solve(G,f)
		elif self.order==0:
			# contribution from dof to the left
			al = (self.grid[ind0][1:]-self.grid[dind]) / 2
			# contribution from dof to the right
			ar = (self.grid[ind0][1:]-self.grid[dind]) / 2
			# forcing term
			f[0]    =                     ar[0]  * ud[0]
			f[1:-1] = al[:-1] * ud[:-1] + ar[1:] * ud[1:]
			f[-1]   = al[-1]  * ud[-1]
			#
			diag = np.zeros((len(ind0),))
			diag[0]    = self.grid[dind][0]  - self.grid[ind0][0]
			diag[1:-1] = self.grid[dind][1:] - self.grid[dind][:-1]
			diag[-1]   = self.grid[ind0][-1] - self.grid[dind][-1]
			# diag[0]    = ar[0]
			# diag[1:-1] = ar[1:] + al[:-1]
			# diag[-1]   = al[-1]
			# print(diag)
			# exit()
			# print(f/diag)
			return f / diag


	# def decompose(self, ind0, dind):
	# 	# detail coefficients
	# 	ud = self.u[dind] - self.interpolate(ind0, dind, self.u[ind0])

	# 	# approximation coefficients
	# 	u0 = self.u[ind0] + self.project(ind0, dind, ud)
	# 	return u0, ud


	# def recompose(self, u0, ud, ind0, dind):
	# 	u = np.zeros_like(self.u)
	# 	u[ind0] = u0 - self.project(ind0, dind, ud)
	# 	u[dind] = self.interpolate(ind0, dind, u[ind0]) + ud
	# 	return u


	def decompose(self, indf, indc, dind):
		'''Compute approximation and detail coefficients at the given level

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		'''
		ind_f = np.ix_(*indf)
		ind_c = np.ix_(*indc)
		

		self.u[ind_c] = self.u_mg[ind_c]
		self.interpolate_nd(indf, indc, dind)

		# detail coefficients
		self.u[ind_c] = 0


		self.u_mg[ind_f] -= self.u[ind_f]

		# approximation coefficients

		self.u[ind_f] = self.u_mg[ind_f]
		#Projection (next 3 lines)
		tc = self.project_nd(indf, indc, dind).copy()
		tc.resize(self.u_mg[ind_c].shape)
		self.u_mg[ind_c] = tc

		return self.u_mg[ind_f]


	def decompose_full(self):
		'''
		  indf: indices of the fine nodes in each dimension
		  indc:	indices of the coarse  nodes in each dimension
		  dind:	indices of the surplus nodes in each dimension
		'''
		self.decompose_grid()

		order = self.order.copy()

		for l in range(len(self.grids[0])):
			indf = [self.grids[d][l][0] for d in range(self.ndim)]
			indc = [self.grids[d][l][1] for d in range(self.ndim)]
			dind = [self.grids[d][l][2] for d in range(self.ndim)]
			if l<=100:
				self.order = self.order2 #[2 for _ in self.order]
			else:
				self.order = order
			# print(self.order,l)
			# # print(indf)
			# print(indc)
			# print(dind)
			# exit()
			self.decompose(indf, indc, dind)
			self.order = order
		return self.u_mg


	def decompose_level(self,level):
		self.decompose_grid()

		order = self.order.copy()
		for l in range(min(level,len(self.grids[0]) ) ):
			indf = [self.grids[d][l][0] for d in range(self.ndim)]
			indc = [self.grids[d][l][1] for d in range(self.ndim)]
			dind = [self.grids[d][l][2] for d in range(self.ndim)]
			if l<=100:
				self.order = self.order2 #[2 for _ in self.order]
			else:
				self.order = order
			# print(self.order,l)
			# # print(indf)
			# print(indc)
			# print(dind)
			# exit()

			self.decompose(indf, indc, dind)
			self.order = order
		return indf,indc,dind,self.u_mg





	def decompose_grid(self):
		'''Decompose original grid into sequence of coarse and surplus grids
		'''

		# self.indf = []
		# self.indc = []
		# self.dind = []
		# for d in range(self.ndim):
		# 	indf_d = np.arange(self.grid[d].size)
		# 	indc_d = indf_d
		# 	min_grid = 3 if self.order==2 else 2
		# 	while len(indc_d)>min_grid:
		# 		dind_d = indc_d[1::2]
		# 		indc_d = indc_d[0::2]
		# 		self.dind.append(dind_d)
		# 		self.indc.append(indc_d)
		# 		self.indf.append(indf_d)
		# return self.indf, self.indc, self.dind

		self.grids = [[] for d in range(self.ndim)]
		for d in range(self.ndim):


			lg = self.grid[d].size #2 ** ( mt.ceil(mt.log2(self.grid[d].size)) ) + 1
			#indf = np.arange(self.grid[d].size)
			indf = np.arange(lg)
			indc = indf.copy()
			min_grid = 3 if (self.order2[d]==2 or self.order[d]==2) else 2
			# min_grid = 2
			while len(indc)>min_grid:
				dind = indc[1::2]
				indc = indc[0::2]
				self.grids[d].append([indf,indc,dind])
				indf = indc

		d_mismatch = np.max( [len(self.grids[d]) for d in range(self.ndim)]   )
		for d in range(self.ndim):
			while len(self.grids[d]) < d_mismatch:
				if len(self.grids[d]) < 1:
					indf= np.arange(self.grid[d].size)
					indc= indf.copy()
				else:
					indf = self.grids[d][-1][0]
					indc = self.grids[d][-1][1]
				self.grids[d].append( [indf, indc, []]  )


	def recompose(self, indf, indc, dind):
		'''Recompose data from approximation and detail coefficients at the given level

		Inputs
		------
		  ind0:	indices of the coarse  nodes
		  dind:	indices of the surplus nodes
		'''
		ind_f = np.ix_(*indf)
		ind_c = np.ix_(*indc)
		

		self.u[ind_f] = self.u_mg[ind_f]

		# approximation coefficients
		tc = self.project_nd(indf, indc, dind).copy()
		tc.resize(self.u_mg[ind_c].shape)
		self.u_mg[ind_c] -= -self.u[ind_c] + tc

		self.u[ind_c] = self.u_mg[ind_c]
		self.interpolate_nd(indf, indc, dind)
		self.u[ind_c] = 0
		self.u_mg[ind_f] += self.u[ind_f]

		# self.u[ind_f] = self.u_mg[ind_f]

		# # detail coefficients
		# self.u_mg[ind_f] -= self.u[ind_f]
		# self.u[ind_c] = 0
		# self.interpolate_nd(indf, indc, dind)

		# # approximation coefficients
		# self.u_mg[ind0] -= self.project(ind0, dind, self.u_mg[dind])

		# # detail coefficients
		# self.u_mg[dind] += self.interpolate(ind0, dind, self.u_mg[ind0])
		return self.u_mg


	def recompose_full(self):
		order = self.order.copy()
		for l in range(len(self.grids[0])-1,-1,-1):
			if l<=100:
				self.order = self.order2 #[2 for _ in self.order]
			else:
				self.order = order
			indf = [self.grids[d][l][0] for d in range(self.ndim)]
			indc = [self.grids[d][l][1] for d in range(self.ndim)]
			dind = [self.grids[d][l][2] for d in range(self.ndim)]
			self.recompose(indf, indc, dind)
			self.order = order
		# for ind0,dind in self.grids[len(self.grids)::-1]:
		# 	self.recompose(ind0,dind)
		return self.u_mg






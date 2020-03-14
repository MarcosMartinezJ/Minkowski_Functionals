'''
Module for computation of minkowski functionals of a binary array

Use: fun(array)
'''

import numpy as np

def _subr1_2D (Lx, Ly, lattice):
	sur = 0
	cur = 0
	eul = 0
	tmp = np.zeros(((Lx+2)*(Ly+2)), dtype=int)
	for jy in range(0,Ly):
		for jx in range(0,Lx):
			i = jx+Lx*jy
			if lattice[i] > 0:
				new_minkos = _free_2D (Lx+2, Ly+2, jx+1, jy+1, tmp)
				tmp[jx+1+(Lx+2)*(jy+1)] = 1
				sur = sur + new_minkos[0]
				cur = cur + new_minkos[1]
				eul = eul + new_minkos[2]
	return [sur, cur, eul]

#end min_subr1_2D



def _free_2D (Lx, Ly, jx, jy, lattice):
	nedges = 0
	nvert = 0
	for i in [-1,1]:
		jxi = jx+i
		jyi = jy+i
		kcl = 1 - lattice[jxi+Lx*jy]
		nedges = nedges +kcl + 1 - lattice[jx+Lx*jyi]
		for j in [-1,1]:
			k4 = Lx*(jy+j)
			nvert = nvert+kcl*(1-lattice[jxi+k4])*(1-lattice[jx+k4])
	surface = 1
	perimeter = -4 + 2*nedges
	euler2D = 1 + -1*nedges + 1*nvert

	return [surface, perimeter, euler2D]

#end min_free_2D



def fun (array):
	'''
	Computes minkowski functionals of input array

	Input: numpy array where 0 = background and value > 0 == foreground
	Output: list of functionals [M0, M1, M2, (M3)]
	'''
	if len(np.shape(array)) == 2:
		Lx = np.shape(array)[1]
		Ly = np.shape(array)[0]
		lattice = np.zeros((Lx*Ly), dtype=int)
		for i in range(0,Lx):
			for j in range(0,Ly):
				lattice[i+j*Lx] = array[j,i]
		return _subr1_2D(Lx, Ly, lattice)

	elif len(np.shape(array)) == 3:
		Lx = np.shape(array)[2]
		Ly = np.shape(array)[1]
		Lz = np.shape(array)[0]
		lattice = np.zeros((Lx*Ly*Lz), dtype=int)
		for i in range(0,Lx):
			for j in range(0,Ly):
				for k in range(0,Lz):
					lattice[i+Lx*(j+Ly*k)] = array[k,j,i]
		return _subr1_3D(Lx, Ly, Lz, lattice)

	else:
		return print("Incorrent array dimensions")

#end fun



def _subr1_3D (Lx, Ly, Lz, lattice):
	vol = 0
	sur = 0
	cur = 0
	eul = 0
	tmp = np.zeros(((Lx+2)*(Ly+2)*(Lz+2)), dtype=int)
	for jz in range(0,Lz):
		for jy in range(0,Ly):
			for jx in range(0,Lx):
				i = jx + Lx*(jy + jz*Ly)
				if lattice[i] > 0:
					new_minkos = _free_3D(Lx+2, Ly+2, Lz+2, jx+1, jy+1, jz+1, tmp)
					tmp[jx+1+(Lx+2)*(jy+1+(Ly+2)*(jz+1))] = 1
					vol = vol + new_minkos[0]
					sur = sur + new_minkos[1]
					cur = cur + new_minkos[2]
					eul = eul + new_minkos[3]
	return [vol, sur, cur, eul]

#end min_subr1_3D



def _free_3D (Lx, Ly, Lz, jx, jy, jz, lattice):
	nfaces = 0
	nedges = 0
	nvert = 0
	for i in [-1,1]:
		jxi = jx+i
		jyi = jy+i
		jzi = jz+i
		kc1 = 1- lattice[jxi + Lx*(jy + Ly*jz)]
		kc2 = 1- lattice[jx + Lx*(jyi + Ly*jz)]
		kc3 = 1- lattice[jx + Lx*(jy + Ly*jzi)]
		nfaces = nfaces + kc1 + kc2 + kc3
		for j in [-1,1]:
			jyj = jy+j
			jzj = jz+j
			k4 = Lx*(jyj+Ly*jz)
			k7 = Lx*(jy+Ly*jzj)
			kc7 = 1- lattice[jx+k7]
			kc1kc4kc5 = kc1*(1-lattice[jxi + k4])*(1-lattice[jx+k4])
			nedges = nedges + kc1kc4kc5 + kc2*(1-lattice[jx+Lx*(jyi+Ly*jzj)])*kc7 + kc1*(1-lattice[jxi + k7])*kc7
			if kc1kc4kc5 != 0:
				for k in [-1,1]:
					jzk = jz+k
					k9 = Lx*(jy+Ly*jzk)
					k10 = Lx*(jyj+Ly*jzk)
					nvert = nvert + (1-lattice[jxi+k9])*(1-lattice[jxi+k10])*(1-lattice[jx+k9])*(1-lattice[jx+k10])
	volume = 1
	surface = (-6) + 2*nfaces
	curvature = 3 + (-2)*nfaces + nedges
	euler = -1 + 1*nfaces + -1*nedges + nvert

	return [volume, surface, curvature, euler]

 #end min_free_3D

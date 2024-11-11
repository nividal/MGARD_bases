import struct
import numpy as np
import os

from codecs import decode
import zstd as zstd
import dahuffman as dhf

#######
# bitplane
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



def bitplane_mask(u,mask):
		## Apply bitmask to all elements of u
		for e,v in np.ndenumerate(u):
				u[e] = bin_to_float(bin(int(float_to_bin(v),2) & mask))
		return u




def bitplanes(u):
	## Return bitplanes
	planes=[]
	for i in range(64):
		planes.append(np.zeros(u.shape,dtype=np.uint8))
		for e,v in np.ndenumerate(u):
			planes[i][e]=np.uint8((int(float_to_bin(v),2) & 2**i) / (2**i))
	return planes


def rec_planes(planes,lo,hi):
	## reconstruct from selected bitplanes
	u=np.zeros(planes[0].shape,dtype=np.uint64)
	for i in range(lo,hi):
		u += planes[i] * (2**i)
	return u

def binarr_to_float(u):
	## binarry array to float array
	u2=np.zeros(u.shape)
	for e,v in np.ndenumerate(u):
		u2[e]=bin_to_float(bin(v))
	return u2




#####
# Compressions
#####

def compress_zstd(u, file='test.npz'):
	if type(u)==np.ndarray:
		data=zstd.ZSTD_compress(u.tobytes())
	elif type(u)== bytes:
		data=zstd.ZSTD_compress(u)
	else:
		print("Eror: unknown type in zstd save")
	np.save(file,data)
	return len(data)


def save_array(u,compression = 'default', file='test'):
		if compression  == 'None':
			np.save(f'{file}.npy',u)
			size = os.stat(f'{file}.npy').st_size
			os.remove(f'{file}.npy')
			return size

		if compression == 'default':
			np.savez_compressed(f'{file}.npz', u.tobytes())
			size=os.stat(f'{file}.npz').st_size
			os.remove(f'{file}.npz')
			return size

		if compression == 'zstd':
			s0 = compress_zstd(u,f'{file}')
			size=os.stat(f'{file}.npy').st_size
			os.remove(f'{file}.npy')
			return s0
		

def huffman_code(u):
	data=u.tobytes()
	codec = dhf.HuffmanCodec.from_data(data)
	return codec,codec.encode(data)

def huffman_decode(data,codec,shape):
	b=codec.decode(data)
	u=np.frombuffer(b, dtype=float)
	u.reshape(shape)
	return u

def save_by_bitplane(u,compression = 'default', file='test'):
	sizes=[]
	for i in range(64):
		mask = 2 ** i
		plane = bitplane_mask(u.copy(),mask)

		if compression == 'default':
			np.savez_compressed(f'{file}_{i}.npz', plane.tobytes())
			size=os.stat(f'{file}_{i}.npz').st_size
			sizes.append(size)
			os.remove(f'{file}_{i}.npz')

		if compression == 'zstd':
			compress_zstd(plane,f'{file}_{i}')
			size=os.stat(f'{file}_{i}.npy').st_size
			sizes.append(size)
			os.remove(f'{file}_{i}.npy')
		
		'''
		if compression == 'huffman':
			sizes.append(compress_huffman(u,f'{file}_{i}.npz'))
		'''
	return sizes

def save_arrays(arrays,compression = 'zstd', file='test'):
	size=[]
	for i in range(len(arrays)):
		if compression == 'default':
			np.savez_compressed(f'{file}_{i}.npz',arrays[i].tobytes())
			size.append(os.stat(f'{file}_{i}.npz').st_size)
			os.remove(f'{file}_{i}.npz')

		if compression == 'zstd':
			size.append(compress_zstd(arrays[i],f'{file}_{i}.npy'))
			os.remove(f'{file}_{i}.npy')
	return size



	##### MGARD workflow
def mgard_external_compression(u,executable="/ccs/home/vidaln/MGARD/install-serial/bin/mgard-cpu",tempfile="data.bin",outputfile="dc.bin"):
	with open(tempfile,"w") as f:
		u.tofile(f,sep="",format='%f')
	p=subprocess.run([executable, "-z", "-i", "data.bin", "-c", "dc.bin", "-t", prec, "-n","2", str(u.shape[0]), str(u.shape[1]),"-m", "abs", "-e", str(error_abs), "-s","0","-l", "2", "-d", "auto"],stdout=stdout_f,stderr=stderr_f,env=env_repl)
	with open(outputfile,"rb") as f:
		u2=np.fromfile(f,dtype="f")
	return u2

def mgard_external_decompression(u,executable="/ccs/home/vidaln/MGARD/install-serial/bin/mgard-cpu",input_file="dc.bin",tempfile="data_dc.bin"):
	with open(tempfile,"w") as f:
		u.tofile(f,sep="",format='%f')
	p=subprocess.run([executable, "-x", "-c", "dc.bin", "-d", "data_dc.bin", "-t", prec, "-n","2", str(u.shape[0]), str(u.shape[1]),"-m", "abs", "-e", str(error_abs), "-s","0","-l", "2", "-d", "auto"],stdout=stdout_f,stderr=stderr_f,env=env_repl)
	with open(tempfile,"rb") as f:
		u2=np.fromfile(f,dtype="f")
	return u2


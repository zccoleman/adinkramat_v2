from abc import abstractmethod, ABCMeta
import sympy as spy
import numpy as np
import array_to_latex as a2l


class Hymn(metaclass=ABCMeta):
	def __init__(self):
		self.b0 = np.diag([1,1,1,1])
		self.b1 = np.diag([-1,1,1,1])
		self.b2 = np.diag([1,-1,1,1])
		self.b3 = np.diag([-1,-1,1,1])
		self.b4 = np.diag([1,1,-1,1])
		self.b5 = np.diag([-1,1,-1,1])
		self.b6 = np.diag([1,-1,-1,1])
		self.b7 = np.diag([-1,-1,-1,1])
		self.b8 = np.diag([1,1,1,-1])
		self.b9 = np.diag([-1,1,1,-1])
		self.b10 = np.diag([1,-1,1,-1])
		self.b11 = np.diag([-1,-1,1,-1])
		self.b12 = np.diag([1,1,-1,-1])
		self.b13 = np.diag([-1,1,-1,-1])
		self.b14 = np.diag([1,-1,-1,-1])
		self.b15 = np.diag([-1,-1,-1,-1])

		self.t     =np.diag([1,1,1,1])
		self.t34   =np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
		self.t23   =np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
		self.t234  =np.array([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0]])
		self.t243  =np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]])
		self.t24   =np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
		self.t12   =np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])

		self.t12_34=np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
		self.t123  =np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]])
		self.t1234 =np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
		self.t1243 =np.array([[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,0,1,0]])
		self.t124  =np.array([[0,1,0,0],[0,0,0,1],[0,0,1,0],[1,0,0,0]])
		self.t132  =np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
		self.t1342 =np.array([[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0]])
		self.t13   =np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
		self.t134  =np.array([[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0]])
		self.t13_24=np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
		self.t1324 =np.array([[0,0,1,0],[0,0,0,1],[0,1,0,0],[1,0,0,0]])
		self.t1432 =np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
		self.t142  =np.array([[0,0,0,1],[1,0,0,0],[0,0,1,0],[0,1,0,0]])
		self.t143  =np.array([[0,0,0,1],[0,1,0,0],[1,0,0,0],[0,0,1,0]])
		self.t14   =np.array([[0,0,0,1],[0,1,0,0],[0,0,1,0],[1,0,0,0]])
		self.t1423 =np.array([[0,0,0,1],[0,0,1,0],[1,0,0,0],[0,1,0,0]])
		self.t14_23=np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])

		self.zeros4=np.zeros((4,4),dtype=int)
		self.zeros8=np.zeros((8,8),dtype=int)

	@staticmethod
	def concater(M11,M12,M21,M22):
		top = np.concatenate((M11,M12),1)
		bottom = np.concatenate((M21,M22),1)
		return np.concatenate((top,bottom),0)

	@abstractmethod
	def check_algebra_LR(self,L_list):
		failed_pairs = []
		R_list = []
		for i in L_list:
			R_list.append(np.transpose(i))

		for i in range(len(L_list)):
			for j in range(len(R_list)):
				if i==j:
					#print(np.dot(L_list[i],R_list[j])+np.dot(L_list[j],R_list[i]))
					#print('algebra not satisfied w/ ' + str(self.class_name))
					#print('__________________________________________________________________________________________________')
					if (np.dot(L_list[i],R_list[j])+np.dot(L_list[j],R_list[i]) == 2*np.identity(8,dtype=int)).all():
						pass
					else:
						print(np.dot(L_list[i],R_list[j])+np.dot(L_list[j],R_list[i]))
						print('algebra not satisfied w/ ' + str(self.class_name) + ' for pair ' + str((i+1,j+1)))
						print('__________________________________________________________________________________________________')
						failed_pairs.append((i+1,j+1))
				else:
					if (np.dot(L_list[i],R_list[j])+np.dot(L_list[j],R_list[i]) == np.zeros((8,8),dtype=int)).all():
						pass
					else:
						print(np.dot(L_list[i],R_list[j])+np.dot(L_list[j],R_list[i]))
						print('algebra not satisfied w/ ' + str(self.class_name) + ' for pair ' + str((i+1,j+1)))
						print('__________________________________________________________________________________________________')
						failed_pairs.append((i+1,j+1))
		if failed_pairs != []:
			print("Pairs that failed algebra LR for " + str(self.class_name) + ":" + str(failed_pairs))
		else:
			print("All pairs passed LR algebraic check for " + str(self.class_name))
		return failed_pairs

	@abstractmethod
	def check_algebra_RL(self,L_list):
		failed_pairs = []
		R_list = []
		for i in L_list:
			R_list.append(np.transpose(i))

		for i in range(len(L_list)):
			for j in range(len(R_list)):
				if i==j:
					if (np.dot(R_list[i],L_list[j])+np.dot(R_list[j],L_list[i]) == 2*np.identity(8,dtype=int)).all():
						pass
					else:
						print(np.dot(R_list[i],L_list[j])+np.dot(R_list[j],L_list[i]))
						print('algebra not satisfied w/ ' + str(self.class_name) + ' for pair ' + str((i+1,j+1)))
						print('__________________________________________________________________________________________________')
						failed_pairs.append((i+1,j+1))
				else:
					if (np.dot(R_list[i],L_list[j])+np.dot(R_list[j],L_list[i]) == np.zeros((8,8),dtype=int)).all():
						pass
					else:
						print(np.dot(R_list[i],L_list[j])+np.dot(R_list[j],L_list[i]))
						print('algebra not satisfied w/ ' + str(self.class_name) + ' for pair ' + str((i+1,j+1)))
						print('__________________________________________________________________________________________________')
						failed_pairs.append((i+1,j+1))

						failed_pairs.append((i+1,j+1))
		if failed_pairs != []:
			print("Pairs that failed algebra RL for " + str(self.class_name) + ":" + str(failed_pairs))
		else:
			print("All pairs passed RL algebraic check for " + str(self.class_name))

		return failed_pairs

	@abstractmethod
	def color_product(self,color_list,to_latex=False):
		color_matrices = []
		for matrix in color_list:
			c_i=self.concater(self.zeros8,matrix,np.transpose(matrix),self.zeros8)
			color_matrices.append(c_i)

		product=np.identity(16,dtype=int)
		for matrix in color_matrices:
			product=np.dot(product,matrix)

		if to_latex:
			a2l.to_ltx(product, frmt = '{:d}', arraytype = 'array')
		else:
			return product

	def hymn_eigenvalues(self):
		product = self.color_product()
		eigenvalues = str(list(np.diag(product)))
		print("HYMN values for " + str(self.class_name) + "are:" + eigenvalues)
		print("_______________________________________________________"
			 +"_______________________________________________________"
			 +"_______________________________________________________")
		return eigenvalues


class ChiralChiral(Hymn):
	def __init__(self):
		print('Initializing ChiralChiral Multiplet...')
		super().__init__()
		self.l1 = self.concater(np.dot(self.b10,self.t243),self.zeros4,self.zeros4,np.dot(self.b10,self.t243))
		self.l2 = self.concater(np.dot(self.b12,self.t123),self.zeros4,self.zeros4,np.dot(self.b12,self.t123))
		self.l3 = self.concater(np.dot(self.b6,self.t134),self.zeros4,self.zeros4,np.dot(self.b6,self.t134))
		self.l4 = self.concater(np.dot(self.b0,self.t142),self.zeros4,self.zeros4,np.dot(self.b0,self.t142))
		self.l5 = self.concater(self.zeros4,np.dot(self.b15,self.t243),np.dot(self.b0,self.t243),self.zeros4)
		self.l6 = self.concater(self.zeros4,np.dot(self.b9,self.t123),np.dot(self.b6,self.t123),self.zeros4)
		self.l7 = self.concater(self.zeros4,np.dot(self.b3,self.t134),np.dot(self.b12,self.t134),self.zeros4)
		self.l8 = self.concater(self.zeros4,np.dot(self.b5,self.t142),np.dot(self.b10,self.t142),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)


class ChiralVector(Hymn):
	def __init__(self):
		print('Initializing ChiralVector Multiplet...')
		super().__init__()
		self.l1 = self.concater(np.dot(self.b10,self.t243),self.zeros4,self.zeros4,np.dot(self.b10,self.t1243))
		self.l2 = self.concater(np.dot(self.b12,self.t123),self.zeros4,self.zeros4,np.dot(self.b12,self.t23))
		self.l3 = self.concater(np.dot(self.b6,self.t134),self.zeros4,self.zeros4,np.dot(self.b0,self.t14))
		self.l4 = self.concater(np.dot(self.b0,self.t142),self.zeros4,self.zeros4,np.dot(self.b6,self.t1342))
		self.l5 = self.concater(self.zeros4,np.dot(self.b2,self.t243),np.dot(self.b13,self.t1243),self.zeros4)
		self.l6 = self.concater(self.zeros4,np.dot(self.b4,self.t123),np.dot(self.b11,self.t23),self.zeros4)
		self.l7 = self.concater(self.zeros4,np.dot(self.b14,self.t134),np.dot(self.b7,self.t14),self.zeros4)
		self.l8 = self.concater(self.zeros4,np.dot(self.b8,self.t142),np.dot(self.b1,self.t1342),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)


class ChiralTensor(Hymn):
	def __init__(self):
		print('Initializing ChiralTensor Multiplet...')
		super().__init__()
		self.l1 = self.concater(np.dot(self.b10,self.t243),self.zeros4,self.zeros4,np.dot(self.b14,self.t234))
		self.l2 = self.concater(np.dot(self.b12,self.t123),self.zeros4,self.zeros4,np.dot(self.b4,self.t124))
		self.l3 = self.concater(np.dot(self.b6,self.t134),self.zeros4,self.zeros4,np.dot(self.b8,self.t132))
		self.l4 = self.concater(np.dot(self.b0,self.t142),self.zeros4,self.zeros4,np.dot(self.b2,self.t143))
		self.l5 = self.concater(self.zeros4,np.dot(self.b11,self.t243),np.dot(self.b0,self.t234),self.zeros4)
		self.l6 = self.concater(self.zeros4,np.dot(self.b13,self.t123),np.dot(self.b10,self.t124),self.zeros4)
		self.l7 = self.concater(self.zeros4,np.dot(self.b7,self.t134),np.dot(self.b6,self.t132),self.zeros4)
		self.l8 = self.concater(self.zeros4,np.dot(self.b1,self.t142),np.dot(self.b12,self.t143),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)

class VectorVector(Hymn):
	def __init__(self):
		print('Initializing VectorVector Multiplet...')
		super().__init__()
		self.ap = spy.Symbol('ap')
		self.am = spy.Symbol('am')
		self.bp = spy.Symbol('bp')
		self.bm = spy.Symbol('bm')
		self.l1 = self.concater(self.bp*np.dot(self.b10,self.t1243),self.zeros4,self.zeros4,self.ap*np.dot(self.b10,self.t1243))
		self.l2 = self.concater(self.bp*np.dot(self.b12,self.t23),self.zeros4,self.zeros4,self.ap*np.dot(self.b12,self.t23))
		self.l3 = self.concater(self.bp*np.dot(self.b0,self.t14),self.zeros4,self.zeros4,self.ap*np.dot(self.b0,self.t14))
		self.l4 = self.concater(self.bp*np.dot(self.b6,self.t1342),self.zeros4,self.zeros4,self.ap*np.dot(self.b6,self.t1342))
		self.l5 = self.concater(self.zeros4,self.bm*np.dot(self.b10,self.t1243),self.am*np.dot(self.b10,self.t1243),self.zeros4)
		self.l6 = self.concater(self.zeros4,self.bm*np.dot(self.b12,self.t23),self.am*np.dot(self.b12,self.t23),self.zeros4)
		self.l7 = self.concater(self.zeros4,self.bm*np.dot(self.b0,self.t14),self.am*np.dot(self.b0,self.t14),self.zeros4)
		self.l8 = self.concater(self.zeros4,self.bm*np.dot(self.b6,self.t1342),self.am*np.dot(self.b6,self.t1342),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)


class TensorTensor(Hymn):
	def __init__(self):
		print('Initializing TensorTensor Multiplet...')
		super().__init__()
		self.ap = spy.Symbol('ap')
		self.am = spy.Symbol('am')
		self.bp = spy.Symbol('bp')
		self.bm = spy.Symbol('bm')
		self.l1 = self.concater(self.bp*np.dot(self.b14,self.t234),self.zeros4,self.zeros4,self.ap*np.dot(self.b14,self.t234))
		self.l2 = self.concater(self.bp*np.dot(self.b4,self.t124),self.zeros4,self.zeros4,self.ap*np.dot(self.b4,self.t124))
		self.l3 = self.concater(self.bp*np.dot(self.b8,self.t132),self.zeros4,self.zeros4,self.ap*np.dot(self.b8,self.t132))
		self.l4 = self.concater(self.bp*np.dot(self.b2,self.t143),self.zeros4,self.zeros4,self.ap*np.dot(self.b2,self.t143))
		self.l5 = self.concater(self.zeros4,self.bm*np.dot(self.b14,self.t234),self.am*np.dot(self.b14,self.t234),self.zeros4)
		self.l6 = self.concater(self.zeros4,self.bm*np.dot(self.b4,self.t124),self.am*np.dot(self.b4,self.t124),self.zeros4)
		self.l7 = self.concater(self.zeros4,self.bm*np.dot(self.b8,self.t132),self.am*np.dot(self.b8,self.t132),self.zeros4)
		self.l8 = self.concater(self.zeros4,self.bm*np.dot(self.b2,self.t143),self.am*np.dot(self.b2,self.t143),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)


class VectorTensor(Hymn):
	def __init__(self):
		print('Initializing VectorTensor Multiplet...')
		super().__init__()
		self.ap = spy.Symbol('ap')
		self.am = spy.Symbol('am')
		self.bp = spy.Symbol('bp')
		self.bm = spy.Symbol('bm')
		self.l1 = self.concater(self.bp*np.dot(self.b14,self.t234),self.zeros4,self.zeros4,self.ap*np.dot(self.b10,self.t1243))
		self.l2 = self.concater(self.bp*np.dot(self.b4,self.t124),self.zeros4,self.zeros4,self.ap*np.dot(self.b12,self.t23))
		self.l3 = self.concater(self.bp*np.dot(self.b8,self.t132),self.zeros4,self.zeros4,self.ap*np.dot(self.b0,self.t14))
		self.l4 = self.concater(self.bp*np.dot(self.b2,self.t143),self.zeros4,self.zeros4,self.ap*np.dot(self.b6,self.t1342))
		self.l5 = self.concater(self.zeros4,self.bm*np.dot(self.b14,self.t234),self.am*np.dot(self.b10,self.t1243),self.zeros4)
		self.l6 = self.concater(self.zeros4,self.bm*np.dot(self.b4,self.t124),self.am*np.dot(self.b12,self.t23),self.zeros4)
		self.l7 = self.concater(self.zeros4,self.bm*np.dot(self.b8,self.t132),self.am*np.dot(self.b0,self.t14),self.zeros4)
		self.l8 = self.concater(self.zeros4,self.bm*np.dot(self.b2,self.t143),self.am*np.dot(self.b6,self.t1342),self.zeros4)
		self.l_list = [self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8]
		self.class_name = __class__

	def color_product(self,to_latex=False):
		return super().color_product(self.l_list,to_latex)

	def check_algebra_LR(self):
		return super().check_algebra_LR(self.l_list)

	def check_algebra_RL(self):
		return super().check_algebra_RL(self.l_list)


if __name__=='__main__':
	CC = ChiralChiral()
	CC.check_algebra_LR()
	CC.check_algebra_RL()
	CC.hymn_eigenvalues()
	CV = ChiralVector()
	CV.check_algebra_LR()
	CV.check_algebra_RL()
	CV.hymn_eigenvalues()
	CT = ChiralTensor()
	CT.check_algebra_LR()
	CT.check_algebra_RL()
	CT.hymn_eigenvalues()
	VV = VectorVector()
	VV.check_algebra_LR()
	VV.check_algebra_RL()
	VV.hymn_eigenvalues()
	TT = TensorTensor()
	TT.check_algebra_LR()
	TT.check_algebra_RL()
	TT.hymn_eigenvalues()
	VT = VectorTensor()
	VT.check_algebra_LR()
	VT.check_algebra_RL()
	VT.hymn_eigenvalues()

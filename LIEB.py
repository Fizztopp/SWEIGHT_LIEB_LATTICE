# -*- coding: utf-8 -*-
"""
@author: Gabriel Topp
"""

import matplotlib.pyplot as plt  
import numpy as np
#import spglib
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
from numpy import random

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 2

mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [16.,4]
mpl.rcParams['text.usetex'] = True

NN = 31 # Number of unit cells along one direction
DIM = 3 # number of orbitals
# Hamiltonian parameters
delta = 1.0
J =  1.0
# Floquet prameters'
AA = 1.0
OMEGA = 1.5

b1 = np.array([1., 0.])/2
b2 = np.array([0., 0.])/2
b3 = np.array([0., 1.])/2

Basis1 = [b1, b2, b3]


def KD(a,b):
    if(a==b):
        return 1.0
    else:
        return 0.0

def a(kx):
    return np.cos(kx/2.)+1j*delta*np.sin(kx/2.)

def dadkx1(kx):
    return (-np.sin(kx/2.)+1j*delta*np.cos(kx/2.))/2.

def dadkx2(kx):
    return (-np.cos(kx/2.)-1j*delta*np.sin(kx/2.))/4.

def dadkx3(kx):
    return (np.sin(kx/2.)-1j*delta*np.cos(kx/2.))/8.

def b(ky):
    return np.cos(ky/2.)+1j*delta*np.sin(ky/2.)

def dbdky1(ky):
    return (-np.sin(ky/2.)+1j*delta*np.cos(ky/2.))/2.

def dbdky2(ky):
    return (-np.cos(ky/2.)-1j*delta*np.sin(ky/2.))/4.

def dbdky3(ky):
    return (np.sin(ky/2.)-1j*delta*np.cos(ky/2.))/8.


def Hk(k):
    kx=k[0]
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 2.*J*a(kx)
    B[1,1] = 0.0
    B[1,0] = 2.*J*np.conj(a(kx))
    B[1,2] = 2.*J*b(ky)
    B[2,1] = 2.*J*np.conj(b(ky))
    return B

def dHdkx1(k):
    kx=k[0]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 2.*J*dadkx1(kx)
    B[1,0] = 2.*J*np.conj(dadkx1(kx))
    B[1,2] = 0.
    B[2,1] = 0.
    return B

def dHdkx2(k):
    kx=k[0]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 2.*J*dadkx2(kx)
    B[1,0] = 2.*J*np.conj(dadkx2(kx))
    B[1,2] = 0.
    B[2,1] = 0.
    return B

def dHdkx3(k):
    kx=k[0]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 2.*J*dadkx3(kx)
    B[1,0] = 2.*J*np.conj(dadkx3(kx))
    B[1,2] = 0.
    B[2,1] = 0.
    return B

def dHdky1(k):
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 0.
    B[1,0] = 0.
    B[1,2] = 2.*J*dbdky1(ky)
    B[2,1] = 2.*J*np.conj(dbdky1(ky))
    return B

def dHdky2(k):
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 0.
    B[1,0] = 0.
    B[1,2] = 2.*J*dbdky2(ky)
    B[2,1] = 2.*J*np.conj(dbdky2(ky))
    return B

def dHdky3(k):
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 0.
    B[1,0] = 0.
    B[1,2] = 2.*J*dbdky3(ky)
    B[2,1] = 2.*J*np.conj(dbdky3(ky))
    return B

def Hk_A(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = Hk(k) + A*(dHdkx1(k)+dHdky1(k)) + 0.5*A**2*(dHdkx2(k) + dHdky2(k))
    return B

def dHk_Adkx1(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = dHdkx1(k) + A*(dHdkx2(k)) + 0.5*A**2*(dHdkx3(k))
    return B

def dHk_Adky1(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = dHdky1(k) + A*(dHdky2(k)) + 0.5*A**2*(dHdky3(k))
    return B
       

def GEOMETRIC_TENSOR(PATH):
    NN = np.size(PATH[:,0])
    B = np.zeros((NN,DIM,2,2),dtype=complex)   
    EIGVALS = np.zeros((NN,DIM))
    J_PARA_X= np.zeros((NN,DIM,DIM),dtype=complex)
    J_PARA_Y= np.zeros((NN,DIM,DIM),dtype=complex)

    for k in range(NN):
        EIGVALS[k,:], v = np.linalg.eigh(Hk(PATH[k]))    
        J_PARA_X[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),dHdkx1(PATH[k])),v)
        J_PARA_Y[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),dHdky1(PATH[k])),v)
        for m in range(DIM):
            B[k,m]=0
            for n in range(DIM):
                if m==n:
                    continue  
                B[k,m,0,0] += J_PARA_X[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,0,1] += J_PARA_X[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,0] += J_PARA_Y[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,1] += J_PARA_Y[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
    return B            


def GEOMETRIC_TENSOR_A(PATH, A):
    NN = np.size(PATH[:,0])
    B = np.zeros((NN,DIM,2,2),dtype=complex)   
    EIGVALS = np.zeros((NN,DIM))
    J_PARA_X= np.zeros((NN,DIM,DIM),dtype=complex)
    J_PARA_Y= np.zeros((NN,DIM,DIM),dtype=complex)

    for k in range(NN):
        EIGVALS[k,:], v = np.linalg.eigh(Hk_A(PATH[k],A))    
        J_PARA_X[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),dHk_Adkx1(PATH[k],A)),v)
        J_PARA_Y[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),dHk_Adky1(PATH[k],A)),v)
        for m in range(DIM):
            B[k,m]=0
            for n in range(DIM):
                if m==n:
                    continue  
                B[k,m,0,0] += J_PARA_X[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,0,1] += J_PARA_X[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,0] += J_PARA_Y[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,1] += J_PARA_Y[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
    return B   


############################################################################### BZ and KPATH
A1 = 2.*np.array([b1[0], b1[1], 0.])                                              
A2 = 2.*np.array([b3[0], b3[1], 0.])                                               
A3 = np.array([0.0, 0.0, 1.0])                                               

# Reciprocal space basis vectors of super cell in 1/m
B1 = 2.*np.pi*np.cross(A2,A3)/np.dot(A1,np.cross(A2,A3))                       
B2 = 2.*np.pi*np.cross(A3,A1)/np.dot(A2,np.cross(A3,A1))                       
B3 = 2.*np.pi*np.cross(A1,A2)/np.dot(A3,np.cross(A1,A2)) 


GAMMA = np.array([0.,0.,0.])
MM = B1/2.+B2/2.
X = B1/2.

num_GM = 100
num_MX = 50
num_XG = 50
N_PATH = num_GM+num_MX+num_XG

# Real space basis vectors of super cell in m
A_BZ = np.linalg.norm(np.cross(B1, B2))
N_B1 = np.linalg.norm(B1)
N_B2 = np.linalg.norm(B2)
dk = N_B1/(NN-1)

## Calculate PATCH
MAT_BZ = np.zeros((NN*NN,3)) 
k0 = np.array([0.,0.,0.])-(B1+B2)/2.

for i in range(NN):
    for j in range(NN):
        MAT_BZ[i+NN*j,:] = k0+i*dk*B1/N_B1+j*dk*B2/N_B2


def k_path():    
    '''
    Calculates high symmetry path points and saves as k_path.npz, #of points =  6*PC.k_num+1     
    '''
    K_PATH = np.array([0.,0.,0.])
    for GM in range(num_GM):
       K_PATH = np.append(K_PATH, K_PATH[-3:]+1/num_GM*(MM-GAMMA))
    for MX in range(num_MX):
       K_PATH = np.append(K_PATH, K_PATH[-3:]+1/num_MX*(X-MM))   
    for XG in range(num_XG-1):
       K_PATH = np.append(K_PATH, K_PATH[-3:]+1/num_XG*(GAMMA-X))  
    K_PATH = np.append(K_PATH, -K_PATH) 
    K_PATH = K_PATH.reshape(int(np.size(K_PATH)/3),3)
    num_kpoints = np.size(K_PATH[:,0])
    #for k in range(num_kpoints):
    #    K_PATH[k,:] = np.dot(M_rotz,K_PATH[k,:]) 
    print("Number of kpoints: " + str(num_kpoints) + " (path)")
    file = open('k_path.dat','w')
    for i in range(num_kpoints):
        for j in range(3):
            file.write("%s " % K_PATH[i][j].real)
        file.write("\n")    
    file.close()
    return K_PATH

KPATH = k_path()

# ## PLOT Points
fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(1, 1)
ax01 = fig1.add_subplot(gs1[0,0],projection='3d')
ax01.set_ylabel(r'kx')
ax01.set_ylabel(r'Energy')
ax01.scatter(B1[0], B1[1], B1[2], c="r", marker="o")
ax01.scatter(B2[0], B2[1], B2[2], c="r", marker="o")
ax01.scatter(MAT_BZ[:,0], MAT_BZ[:,1], c="b", marker="o")
ax01.scatter(KPATH[:,0], KPATH[:,1], c="k", marker="o")
plt.show()

# =============================================================================
# 
# Z_PATH = np.zeros((N_PATH,3))
# Z_BZ = np.zeros((NN*NN,3))
# 
# for k in range(N_PATH):   
#     Z_PATH[k,:], v = np.linalg.eigh(Hk(KPATH[k,:], JJ, DELTA))    
# 
# 
# fig2 = plt.figure(2)
# gs2 = gridspec.GridSpec(1, 1)
# ax11 = fig2.add_subplot(gs2[0,0])
# ax11.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
# ax11.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
# ax11.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
# ax11.plot(Z_PATH[:,0], color='b', linewidth=2.0, alpha=1.0)
# ax11.plot(Z_PATH[:,1], color='r', linewidth=2.0, alpha=1.0)
# ax11.plot(Z_PATH[:,2], color='g', linewidth=2.0, alpha=1.0)
# 
# plt.tight_layout()
# plt.show()
# 
# =============================================================================

# =============================================================================
# for k in range(NN*NN): 
#     Z_BZ[k,:], v = np.linalg.eigh(Hk(MAT_BZ[k,:], JJ, DELTA))
# 
# # ## PLOT Points
# fig3 = plt.figure(3)
# gs3 = gridspec.GridSpec(1, 1)
# ax03 = fig3.add_subplot(gs3[0,0],projection='3d')
# ax03.set_ylabel(r'kx')
# ax03.set_ylabel(r'Energy')
# ax03.scatter(MAT_BZ[:,0],MAT_BZ[:,1],Z_BZ[:,0], c="b", marker="o")
# ax03.scatter(MAT_BZ[:,0],MAT_BZ[:,1],Z_BZ[:,1], c="r", marker="o")
# ax03.scatter(MAT_BZ[:,0],MAT_BZ[:,1],Z_BZ[:,2], c="b", marker="o")
# plt.show()
# =============================================================================

#%% ########################################################################## STATIC PROBLEM
N_PATH = np.size(KPATH[:,0])
EIGVALS = np.zeros((N_PATH ,DIM))
JP_DIAG = np.zeros((N_PATH,DIM,DIM))
JD_DIAG = np.zeros((N_PATH,DIM,DIM))

# Light-coupled Hamiltonian
H_A_DIAG = np.zeros((N_PATH,DIM,DIM),dtype=complex)

Metric = np.real(GEOMETRIC_TENSOR(KPATH))
Metric_A = np.real(GEOMETRIC_TENSOR_A(KPATH,AA))

Curv = -2.*np.imag(GEOMETRIC_TENSOR(KPATH))
Curv_A = -2.*np.imag(GEOMETRIC_TENSOR_A(KPATH,AA))

for k in range(N_PATH):   
    EIGVALS[k,:], v = np.linalg.eigh(Hk(KPATH[k]))    
    JP_DIAG[k,:,:] = np.abs(np.dot(np.dot(np.transpose(np.conj(v)),dHdkx1(KPATH[k])),v))+np.abs(np.dot(np.dot(np.transpose(np.conj(v)),dHdky1(KPATH[k])),v))
    JD_DIAG[k,:,:] = np.abs(np.dot(np.dot(np.transpose(np.conj(v)),dHdkx2(KPATH[k])),v))+np.abs(np.dot(np.dot(np.transpose(np.conj(v)),dHdky2(KPATH[k])),v))
    H_A_DIAG[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),Hk_A(KPATH[k],AA)),v)

ABS_PATH = np.zeros(np.size(KPATH[:,0]))
PATH = np.zeros((N_PATH-1,2))
PATH[:,0] = np.diff(KPATH[:,0])
PATH[:,1] = np.diff(KPATH[:,1])

## FULL derivative
dk = np.zeros(N_PATH-1)
for k in range(N_PATH-1):
    dk[k] = np.linalg.norm(PATH[k,:])
    
dE0 = np.diff(EIGVALS[:,0])
dE1 = np.diff(EIGVALS[:,1])
dE2 = np.diff(EIGVALS[:,2])

v0 = dE0/dk
v1 = dE1/dk
v2 = dE2/dk

dvdk0 = np.diff(v0)/dk[0:N_PATH-2]
dvdk1 = np.diff(v1)/dk[0:N_PATH-2]
dvdk2 = np.diff(v2)/dk[0:N_PATH-2]

fig3 = plt.figure(3)
gs3 = gridspec.GridSpec(2, 3)
ax11 = fig3.add_subplot(gs3[0,0])
ax11.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax11.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax11.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax11.plot(EIGVALS[:,0], 'b', linewidth=2.0, alpha=1.0)
ax11.plot(EIGVALS[:,1], 'r', linewidth=2.0, alpha=1.0)
ax11.plot(EIGVALS[:,2], 'g', linewidth=2.0, alpha=1.0)
ax11.plot(np.real(H_A_DIAG[:,0,0]), 'b--', linewidth=2.0, alpha=1.0,label=r'$\mathrm{\epsilon_i(A='+str(AA)+')}$')
ax11.plot(np.real(H_A_DIAG[:,1,1]), 'r--', linewidth=2.0, alpha=1.0)
ax11.plot(np.real(H_A_DIAG[:,2,2]), 'g--', linewidth=2.0, alpha=1.0)

ax12 = fig3.add_subplot(gs3[0,1])
ax12.set_ylabel(r'$\mathrm{\partial\epsilon /\partial k}$ $\mathrm{(eV a)}$')
ax12.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax12.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
#ax12.plot(JP_DIAG[:,0,0], 'b', linewidth=2.0, alpha=1.0)
ax12.plot(JP_DIAG[:,1,1], 'r', linewidth=2.0, alpha=1.0)
#ax12.plot(JP_DIAG[:,2,2], 'g', linewidth=2.0, alpha=1.0)
ax12.plot(v0, 'b--', linewidth=2.0, alpha=1.0)
ax12.plot(v1, 'r--', linewidth=2.0, alpha=1.0)
ax12.plot(v2, 'g--', linewidth=2.0, alpha=1.0)

ax13 = fig3.add_subplot(gs3[0,2])
ax13.set_ylabel(r'$\mathrm{\partial^2\epsilon /\partial k^2}$ $\mathrm{(eV^2 a^2)}$')
ax13.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax13.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
#ax12.plot(JD_DIAG[:,0,0], 'b', linewidth=2.0, alpha=1.0)
ax13.plot(JD_DIAG[:,1,1], 'r', linewidth=2.0, alpha=1.0)
#ax12.plot(JD_DIAG[:,2,2], 'g', linewidth=2.0, alpha=1.0)
ax13.plot(dvdk0, 'b--', linewidth=2.0, alpha=1.0, label=r'$\mathrm{trivial}$')
ax13.plot(dvdk1, 'r--', linewidth=2.0, alpha=1.0)
ax13.plot(dvdk2, 'g--', linewidth=2.0, alpha=1.0)
plt.legend(loc='lower right')

ax14 = fig3.add_subplot(gs3[1,0])
ax14.set_ylabel(r'$\mathrm{Metric}$ $\mathrm{(a^2)}$')
ax14.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax14.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
#ax14.plot(Metric[:,0,0,0], 'b')
ax14.plot(Metric[:,1,0,0], 'r')
##ax14.plot(Metric[:,2,0,0], 'g', label=r'$\mathrm{M_{00}}$')
#ax14.plot(Metric[:,0,0,1], 'b--')
ax14.plot(Metric[:,1,0,1], 'r--')
#ax14.plot(Metric[:,2,0,1], 'g--', label=r'$\mathrm{M_{01}}$')
plt.legend(loc='upper right')

ax15 = fig3.add_subplot(gs3[1,1])
ax15.set_ylabel(r'$\mathrm{BC}$ $\mathrm{(a^2)}$')
ax15.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax15.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
#ax15.plot(Curv[:,0,0,1], 'b--')
ax15.plot(Curv[:,1,0,1], 'r--')
#ax15.plot(Curv[:,2,0,1], 'g--', label=r'$\mathrm{M_{01}}$')
plt.legend(loc='upper right')
#ax3.plot(Metric[:,1])
#ax14.plot(Metric_A[:,0], 'b--')
#ax3.plot(Metric_A[:,1])

plt.tight_layout()
plt.show()


#%% FLOQUET
            
def H_EFF(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = Hk(k) + 0.25*A**2*(dHdkx2(k) + dHdky2(k))
    for m in range(DIM):
        for n in range(DIM):
            for l in range(DIM):
                B[m,n] += 1j*A**2./(2.*OMEGA)*(dHdkx1(k)[m,l]*dHdky1(k)[l,n]-dHdky1(k)[m,l]*dHdkx1(k)[l,n])
    return B

def H_EFFdkx1(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = dHdkx1(k) + 0.25*A**2*(dHdkx3(k))
    return B

def H_EFFdky1(k, A):
    B = np.zeros((DIM,DIM),dtype=complex)
    B = dHdky1(k) + 0.25*A**2*(dHdky3(k))
    return B

def GEOMETRIC_TENSOR_EFF(PATH, A):
    NN = np.size(PATH[:,0])
    B = np.zeros((NN,DIM,2,2),dtype=complex)   
    EIGVALS = np.zeros((NN,DIM))
    J_PARA_X= np.zeros((NN,DIM,DIM),dtype=complex)
    J_PARA_Y= np.zeros((NN,DIM,DIM),dtype=complex)

    for k in range(NN):
        EIGVALS[k,:], v = np.linalg.eigh(H_EFF(PATH[k],A))    
        J_PARA_X[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),H_EFFdkx1(PATH[k],A)),v)
        J_PARA_Y[k,:,:] = np.dot(np.dot(np.transpose(np.conj(v)),H_EFFdky1(PATH[k],A)),v)
        for m in range(DIM):
            B[k,m]=0
            for n in range(DIM):
                if m==n:
                    continue  
                B[k,m,0,0] += J_PARA_X[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,0,1] += J_PARA_X[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,0] += J_PARA_Y[k,m,n]*J_PARA_X[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
                B[k,m,1,1] += J_PARA_Y[k,m,n]*J_PARA_Y[k,n,m]/(EIGVALS[k,m]-EIGVALS[k,n])**2
    return B  

EVALS_FLOQ = np.zeros((N_PATH,DIM))
for k in range(N_PATH):   
    EVALS_FLOQ[k,:], v = np.linalg.eigh(H_EFF(KPATH[k], AA))   

Metric_EFF = np.real(GEOMETRIC_TENSOR_EFF(KPATH, AA))  
Curv_EFF = -2.0*np.imag(GEOMETRIC_TENSOR_EFF(KPATH, AA))  

fig4 = plt.figure(4)
gs4 = gridspec.GridSpec(1, 4)
ax21 = fig4.add_subplot(gs4[0,0])
ax21.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax21.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax21.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax21.plot(EVALS_FLOQ[:,0], 'b', linewidth=2.0, alpha=1.0)
ax21.plot(EVALS_FLOQ[:,1], 'r', linewidth=2.0, alpha=1.0, label=r'$\mathrm{H_{eff}}$')
ax21.plot(EVALS_FLOQ[:,2], 'g', linewidth=2.0, alpha=1.0)
ax21.plot(EIGVALS[:,0], 'b--', linewidth=2.0, alpha=1.0)
ax21.plot(EIGVALS[:,1], 'k--', linewidth=2.0, alpha=1.0, label=r'$\mathrm{H_0}$')
ax21.plot(EIGVALS[:,2], 'g--', linewidth=2.0, alpha=1.0)
plt.legend(loc='upper right')

ax22 = fig4.add_subplot(gs4[0,1])
ax22.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax22.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax22.set_ylabel(r'$\mathrm{M_{00}}$ $\mathrm{(a^2)}$')
ax22.plot(Metric_EFF[:,1,0,0], 'r', label=r'$\mathrm{H_{eff}}$')
ax22.plot(Metric[:,1,0,0], 'k--', label=r'$\mathrm{H_0}$')
#ax3.plot(Metric[:,1])

#ax3.plot(Metric_A[:,1])
plt.legend()

ax23 = fig4.add_subplot(gs4[0,2])
ax23.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax23.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax23.plot(Metric_EFF[:,1,0,1], 'r', label=r'$\mathrm{H_{eff}}$')
ax23.set_ylabel(r'$\mathrm{M_{01}}$ $\mathrm{(a^2)}$')
ax23.plot(Metric[:,1,0,1], 'k--', label=r'$\mathrm{H_0}$')
plt.legend()


ax24 = fig4.add_subplot(gs4[0,3])
ax24.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax24.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax24.set_ylabel(r'$\mathrm{BC}$ $\mathrm{(a^2)}$')
ax24.plot(Curv_EFF[:,1,0,1], 'r', label=r'$\mathrm{H_{eff}}$')
ax24.plot(Curv[:,1,0,1], 'k--', label=r'$\mathrm{H_0}$')
plt.legend()


plt.tight_layout()
plt.show()

#%% Integrated qunatities

print("Are of BZ: "+str(A_BZ))   

QUANTUM_TENSOR = GEOMETRIC_TENSOR(MAT_BZ)[:,0,:,:]
METRIC_BZ = np.real(QUANTUM_TENSOR)
CURV_BZ = -2.*np.imag(QUANTUM_TENSOR)

CHERN = np.sum(CURV_BZ[:,0,1]*A_BZ/NN**2)*0.5/np.pi
print("Chern number (H_0): "+str(CHERN))

INT_M = np.sum(METRIC_BZ[:,0,0]*A_BZ/NN**2)*0.5/np.pi
print("Integrated metric (H_0): "+str(INT_M))

# =============================================================================
# # ## PLOT Points
# fig5 = plt.figure(5,figsize=(8, 6))
# gs5 = gridspec.GridSpec(1, 2)
# ax51 = fig5.add_subplot(gs5[0,0],projection='3d')
# ax51.set_ylabel(r'$\mathrm{M_{00}}$ $\mathrm{(a^2)}$')
# ax51.scatter(MAT_BZ[:,0],MAT_BZ[:,1],METRIC_BZ[:,0,0], c="b", marker="o")
# 
# ax52 = fig5.add_subplot(gs5[0,1],projection='3d')
# ax52.set_ylabel(r'$\mathrm{BC}$ $\mathrm{(a^2)}$')
# ax52.scatter(MAT_BZ[:,0],MAT_BZ[:,1],CURV_BZ[:,0,1], c="b", marker="o")
# plt.show()      
# =============================================================================



QUANTUM_TENSOR_EFF = GEOMETRIC_TENSOR_EFF(MAT_BZ, AA)[:,0,:,:]
METRIC_BZ_EFF = np.real(QUANTUM_TENSOR_EFF)
CURV_BZ_EFF = -2.*np.imag(QUANTUM_TENSOR_EFF)

CHERN_EFF = np.sum(CURV_BZ_EFF[:,0,1]*A_BZ/NN**2)*0.5/np.pi
print("Chern number (H_EFF): "+str(CHERN_EFF))

INT_M = np.sum(METRIC_BZ_EFF[:,0,0]*A_BZ/NN**2)*0.5/np.pi
print("Integrated metric (H_0): "+str(INT_M))

# =============================================================================
# # ## PLOT Points
# fig6 = plt.figure(6,figsize=(8, 6))
# gs6 = gridspec.GridSpec(1, 2)
# ax51 = fig5.add_subplot(gs6[0,0],projection='3d')
# ax51.set_ylabel(r'$\mathrm{M_{00}}$ $\mathrm{(a^2)}$')
# ax51.scatter(MAT_BZ[:,0],MAT_BZ[:,1],METRIC_BZ_EFF[:,0,0], c="b", marker="o")
# 
# ax62 = fig6.add_subplot(gs6[0,1],projection='3d')
# ax62.set_ylabel(r'$\mathrm{BC}$ $\mathrm{(a^2)}$')
# ax62.scatter(MAT_BZ[:,0],MAT_BZ[:,1],CURV_BZ_EFF[:,0,1], c="b", marker="o")
# plt.show()       
# =============================================================================


#%% BdG Hamiltonian
UU = -10*J
dev_mu = 1e-6
calibrate = 1e-3
nu = 1.5
dev_order = 1e-5 
TT = 0.005*J
max_iter = 25

def fermi(energy):
    return 1./(np.exp(energy/TT) + 1.)

def fermi_zero(energy):
    if(-1e-5<energy<+1e-5):
        return 0.5
    elif(energy>=1e-5): 
        return 0.0
    else:
        return 1.0

def fermi_SP(energy,mu):
    return 1./(np.exp((energy-mu)/TT) + 1.)

def set_mu_SP(KPATH):
    dev = 1.0
    mu = random.rand()
    EVALS = np.zeros((NN**2,DIM))
    for k in range(NN**2):
        ev, vec = np.linalg.eigh(Hk(KPATH[k])) 
        EVALS[k,:] = ev
        
    while(dev>dev_mu):
        mu_old = mu
        N_tot = 0.
        for k in range(NN**2):
            for i in range(DIM):
                N_tot += fermi_SP(EVALS[k,i],mu)
        mu += -calibrate*(N_tot/NN**2-nu)        
        dev = np.abs(mu-mu_old)
    print("dev = "+str(dev))
    print("mu = "+str(mu))    
    return mu    

def set_ORDER(RHO0):
    OP = np.zeros((3,DIM),dtype=complex)
    for i in range(DIM):
        OP[0,i] = UU/NN**2*RHO0[i,i+DIM]
        OP[1,i] = 1./NN**2*RHO0[i,i]
        OP[2,i] = 1./NN**2*RHO0[DIM+i,DIM+i]
    return OP

def set_HK_BdG(mu,k,OP): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_UP = Hk(k)
    HK_DOWN = Hk(-k)
   
    for i in range(DIM):
        HK_BdG[0,1,i,i] = OP[0,i]
        HK_BdG[1,0,i,i] = np.conj(OP[0,i])
    
    HK_BdG[0,0,:,:] = HK_UP - np.eye(DIM)*mu
    HK_BdG[1,1,:,:] = -np.conj(HK_DOWN) + np.eye(DIM)*mu
    return HK_BdG
  

def dewrap(ARRAY):
    ARRAY_EXPAND = np.zeros((2*DIM,2*DIM),dtype=complex)
    for m in range(2):
        for n in range(2):
            for i in range(DIM):
                for j in range(DIM):
                    ARRAY_EXPAND[m*DIM+i,n*DIM+j] = ARRAY[m,n,i,j] 
    return ARRAY_EXPAND                
   
def Bands_BdG(mu,KPATH,OP):
    NPATH = np.size(KPATH[:,0])
    BANDS = np.zeros((NPATH,2*DIM))
    for k in range(NPATH):
        H = set_HK_BdG(mu,KPATH[k],OP)  
        ev, vec = np.linalg.eigh(dewrap(H))   
        BANDS[k,:] = ev
    return BANDS  

def GS_BdG(mu,KPATH):
    dev = 1.0
    count = 0
    ORDER = np.zeros((3,DIM),dtype=complex) 
    RHO = np.zeros((2*DIM,2*DIM),dtype=complex)
    #Set initial order parameter
    for i in range(DIM):
        re = random.rand()
        im = random.rand()
        ORDER[0,i] = re + 1j*im
    #Calculate order parameter
    while(dev > dev_order):
        print(count)
        ORDER_OLD = ORDER
        ORDER = np.zeros((3,DIM),dtype=complex)
        for k in range(NN**2):
            H=set_HK_BdG(mu,KPATH[k],ORDER_OLD) 
            ev, vec = np.linalg.eigh(dewrap(H))  
            for i in range(2*DIM):
                for j in range(2*DIM):
                    RHO[i,j] = fermi(ev[i])*KD(i,j) 
            RHO = np.dot(vec,np.dot(RHO,np.transpose(np.conj(vec))))
            ORDER += set_ORDER(RHO) 
        #calc deviation
        dev = np.amax(np.abs(ORDER)-np.abs(ORDER_OLD))  
        print("amax(ORDER): "+str(np.amax(np.abs(ORDER[0,:]))))
        print("deviation: "+str(dev))
        count=count+1 
        if(count > max_iter):
            break
    return ORDER    
        
MU = set_mu_SP(MAT_BZ)   
ORDER = GS_BdG(MU,MAT_BZ) 
print("Order parameter: "+str(ORDER[0,:]/J)) 
EIGVALS_BdG = Bands_BdG(MU,KPATH, ORDER)   
 
fig3 = plt.figure(3)
gs3 = gridspec.GridSpec(1, 1)
ax3 = fig3.add_subplot(gs3[0,0])
ax3.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax3.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax3.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax3.plot(EIGVALS_BdG[:,:], 'k', linewidth=2.0, alpha=1.0)   
plt.plot()

#%% BdG Superfluid weight

eps = 1e-6

def dfermi(energy):
    temp = energy/TT
    if(np.abs(temp)>100.):
        temp = temp*100./np.abs(temp)
    return -np.exp(temp)/(TT*(np.exp(temp) + 1.)**2)

# =============================================================================
# def dHdky1s(k):
#     kx=k[0]
#     ky=k[1]
#     B = np.zeros((DIM,DIM),dtype=complex)
#     B[0,1] = -2.*J*(np.cos(-kx/2.)-1j*delta*np.sin(-kx/2.))
#     B[1,1] = 0.0
#     B[1,0] = -2.*J*(np.cos(-kx/2.)+1j*delta*np.sin(-kx/2.))
#     B[1,2] = -2.*J*(np.cos(-ky/2.)-1j*delta*np.sin(-ky/2.))
#     B[2,1] = -2.*J*(np.cos(-ky/2.)+1j*delta*np.sin(-ky/2.))
#     return B
# =============================================================================

def dHdkx1s(k):
    kx=k[0]
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = 2.*J*(np.sin(-kx/2.))*(-1/2)
    B[1,1] = 0.0
    B[1,0] = 2.*J*(np.sin(-kx/2.))*(-1/2)
    B[1,2] = 2.*J*(np.sin(-ky/2.))*(-1/2)
    B[2,1] = 2.*J*(np.sin(-ky/2.))*(-1/2)
    return B

def dHdky1s(k):
    kx=k[0]
    ky=k[1]
    B = np.zeros((DIM,DIM),dtype=complex)
    B[0,1] = -2.*J*(-1j*delta*np.cos(-kx/2.))*(-1/2)
    B[1,1] = 0.0
    B[1,0] = -2.*J*(+1j*delta*np.cos(-kx/2.))*(-1/2)
    B[1,2] = -2.*J*(-1j*delta*np.cos(-ky/2.))*(-1/2)
    B[2,1] = -2.*J*(+1j*delta*np.cos(-ky/2.))*(-1/2)
    return B

def set_dHK_BdGdkx(k): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_BdG[0,0,:,:] = dHdkx1(k)
    HK_BdG[1,1,:,:] = dHdkx1s(k)
    return HK_BdG

def set_dHK_BdGdky(k): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_BdG[0,0,:,:] = dHdky1(k)
    HK_BdG[1,1,:,:] = dHdkx1s(k)
    return HK_BdG

def SF_weight_full(mu,KPATH,OP):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG(mu,KPATH[k],OP))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = dewrap(set_dHK_BdGdkx(KPATH[k]))
        dHdky = dewrap(set_dHK_BdGdky(KPATH[k]))
        
        # transform to BdG band basis
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])  
    return WEIGHT                


def SF_weight_trivial(mu,KPATH,OP):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG(mu,KPATH[k],OP))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = set_dHK_BdGdkx(KPATH[k])
        dHdky = set_dHK_BdGdky(KPATH[k])
        
        ### remove geoemtric (interband) part
        ev_BLOCH_p, vec_BLOCH_p = np.linalg.eigh(Hk(KPATH[k]))                 # eigensystem of particle Hamiltonian (Bloch functions)
        ev_BLOCH_h, vec_BLOCH_h = np.linalg.eigh(-np.conj(Hk(-KPATH[k])))      # eigensystem of hole Hamiltonian
        
        # transform current into Bloch basis
        dHdkx[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdkx[0,0,:,:]),vec_BLOCH_p)
        dHdkx[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdkx[1,1,:,:]),vec_BLOCH_h)
        dHdky[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdky[0,0,:,:]),vec_BLOCH_p)
        dHdky[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdky[1,1,:,:]),vec_BLOCH_h)
        # delete trivial intraband part
        for m in range(DIM):
            for n in range(DIM):
                dHdkx[0,0,m,n] = KD(m,n)*dHdkx[0,0,m,n]         
                dHdkx[1,1,m,n] = KD(m,n)*dHdkx[1,1,m,n]           
                dHdky[0,0,m,n] = KD(m,n)*dHdky[0,0,m,n]         
                dHdky[1,1,m,n] = KD(m,n)*dHdky[1,1,m,n]                                                        
        # transoform back to k-orbital baisis
        dHdkx[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdkx[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdkx[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdkx[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        dHdky[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdky[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdky[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdky[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        ###
        
        # transform to BdG band basis
        dHdkx = dewrap(dHdkx)
        dHdky = dewrap(dHdky)
        
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])   
    return WEIGHT      


def SF_weight_geom(mu,KPATH,OP):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG(mu,KPATH[k],OP))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = set_dHK_BdGdkx(KPATH[k])
        dHdky = set_dHK_BdGdky(KPATH[k])
        
        ### remove geoemtric (interband) part
        ev_BLOCH_p, vec_BLOCH_p = np.linalg.eigh(Hk(KPATH[k]))                 # eigensystem of particle Hamiltonian (Bloch functions)
        ev_BLOCH_h, vec_BLOCH_h = np.linalg.eigh(-np.conj(Hk(-KPATH[k])))      # eigensystem of hole Hamiltonian
        
        # transform current into Bloch basis
        dHdkx[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdkx[0,0,:,:]),vec_BLOCH_p)
        dHdkx[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdkx[1,1,:,:]),vec_BLOCH_h)
        dHdky[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdky[0,0,:,:]),vec_BLOCH_p)
        dHdky[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdky[1,1,:,:]),vec_BLOCH_h)
        # delete trivial intraband part
        for m in range(DIM):
            dHdkx[0,0,m,m] = 0.       
            dHdkx[1,1,m,m] = 0.         
            dHdky[0,0,m,m] = 0.          
            dHdky[1,1,m,m] = 0.                                                         
        # transoform back to k-orbital baisis
        dHdkx[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdkx[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdkx[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdkx[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        dHdky[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdky[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdky[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdky[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        ###
        
        # transform to BdG band basis
        dHdkx = dewrap(dHdkx)
        dHdky = dewrap(dHdky)
        
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])   
    return WEIGHT                     

DD = SF_weight_full(MU,MAT_BZ,ORDER)
DD_TRIV = SF_weight_trivial(MU,MAT_BZ,ORDER)
DD_GEO = SF_weight_geom(MU,MAT_BZ,ORDER)
print("weight: "+"\n"+str(DD)+"\n")
print("weight trivial: "+"\n"+str(DD_TRIV)+"\n")
print("weight geometric: "+"\n"+str(DD_GEO)+"\n")

#%% BdG Hamiltonian (effective Floquet Hamiltonian)


def set_mu_SP_EFF(KPATH, A):
    dev = 1.0
    mu = random.rand()
    EVALS = np.zeros((NN**2,DIM))
    for k in range(NN**2):
        ev, vec = np.linalg.eigh(H_EFF(KPATH[k],A)) 
        EVALS[k,:] = ev
        
    while(dev>dev_mu):
        mu_old = mu
        N_tot = 0.
        for k in range(NN**2):
            for i in range(DIM):
                N_tot += fermi_SP(EVALS[k,i],mu)
        mu += -calibrate*(N_tot/NN**2-nu)        
        dev = np.abs(mu-mu_old)
    print("dev = "+str(dev))
    print("mu = "+str(mu))    
    return mu    


def set_HK_BdG_EFF(mu, k, OP, A): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_UP = H_EFF(k,A)
    HK_DOWN = H_EFF(-k,A)
   
    for i in range(DIM):
        HK_BdG[0,1,i,i] = OP[0,i]
        HK_BdG[1,0,i,i] = np.conj(OP[0,i])
    
    HK_BdG[0,0,:,:] = HK_UP - np.eye(DIM)*mu
    HK_BdG[1,1,:,:] = -np.conj(HK_DOWN) + np.eye(DIM)*mu
    return HK_BdG
               
   
def Bands_BdG_EFF(mu, KPATH, OP, A):
    NPATH = np.size(KPATH[:,0])
    BANDS = np.zeros((NPATH,2*DIM))
    for k in range(NPATH):
        H = set_HK_BdG_EFF(mu,KPATH[k],OP,A)  
        ev, vec = np.linalg.eigh(dewrap(H))   
        BANDS[k,:] = ev
    return BANDS  


def GS_BdG_EFF(mu, KPATH, A):
    dev = 1.0
    count = 0
    ORDER = np.zeros((3,DIM),dtype=complex) 
    RHO = np.zeros((2*DIM,2*DIM),dtype=complex)
    #Set initial order parameter
    for i in range(DIM):
        re = random.rand()
        im = random.rand()
        ORDER[0,i] = re + 1j*im
    #Calculate order parameter
    while(dev > dev_order):
        print(count)
        ORDER_OLD = ORDER
        ORDER = np.zeros((3,DIM),dtype=complex)
        for k in range(NN**2):
            H=set_HK_BdG_EFF(mu,KPATH[k],ORDER_OLD,A) 
            ev, vec = np.linalg.eigh(dewrap(H))  
            for i in range(2*DIM):
                for j in range(2*DIM):
                    RHO[i,j] = fermi(ev[i])*KD(i,j) 
            RHO = np.dot(vec,np.dot(RHO,np.transpose(np.conj(vec))))
            ORDER += set_ORDER(RHO) 
        #calc deviation
        dev = np.amax(np.abs(ORDER-ORDER_OLD))    
        print("amax(ORDER): "+str(np.amax(np.abs(ORDER[0,:]))))
        print("deviation: "+str(dev))
        count=count+1 
        if(count > max_iter):
            break
    return ORDER    
        

MU_EFF= set_mu_SP_EFF(MAT_BZ,AA)   
ORDER_EFF = GS_BdG_EFF(MU_EFF,MAT_BZ,AA) 
print("Order parameter: "+str(ORDER_EFF[0,:])) 
EIGVALS_BdG_EFF = Bands_BdG_EFF(MU_EFF, KPATH, ORDER_EFF, AA)   
 
fig4 = plt.figure(3)
gs4 = gridspec.GridSpec(1, 1)
ax4 = fig4.add_subplot(gs4[0,0])
ax4.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax4.set_xticks([0 , num_GM, num_GM+num_MX, num_GM+num_MX+num_XG])
ax4.set_xticklabels([r'$\mathrm{\Gamma}$', r'$\mathrm{M}$' , r'$\mathrm{X}$', '$\mathrm{\Gamma}$'])
ax4.plot(EIGVALS_BdG_EFF[:,:], 'k', linewidth=2.0, alpha=1.0)   
plt.plot()

#%% BdG Superfluid weight (effective Flqouet Hamiltonian)

eps = 1e-6

def set_dHK_BdGdkx_EFF(k, A): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_BdG[0,0,:,:] = H_EFFdkx1(k,A)
    HK_BdG[1,1,:,:] = -np.conj(H_EFFdkx1(-k, A))
    return HK_BdG

def set_dHK_BdGdky_EFF(k, A): 
    HK_BdG = np.zeros((2,2,DIM,DIM),dtype=complex)
    HK_BdG[0,0,:,:] = H_EFFdky1(k,A)
    HK_BdG[1,1,:,:] = -np.conj(H_EFFdky1(-k,A))
    return HK_BdG

def SF_weight_full_EFF(mu, KPATH, OP, A):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG_EFF(mu,KPATH[k],OP,A))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = dewrap(set_dHK_BdGdkx_EFF(KPATH[k],A))
        dHdky = dewrap(set_dHK_BdGdky_EFF(KPATH[k],A))
        
        # transform to BdG band basis
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])  
    return WEIGHT                


def SF_weight_trivial_EFF(mu, KPATH, OP, A):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG_EFF(mu,KPATH[k],OP,A))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = set_dHK_BdGdkx_EFF(KPATH[k],A)
        dHdky = set_dHK_BdGdky_EFF(KPATH[k],A)
        
        ### remove geoemtric (interband) part
        ev_BLOCH_p, vec_BLOCH_p = np.linalg.eigh(H_EFF(KPATH[k],A))                 # eigensystem of particle Hamiltonian (Bloch functions)
        ev_BLOCH_h, vec_BLOCH_h = np.linalg.eigh(-np.conj(H_EFF(-KPATH[k],A)))      # eigensystem of hole Hamiltonian
        
        # transform current into Bloch basis
        dHdkx[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdkx[0,0,:,:]),vec_BLOCH_p)
        dHdkx[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdkx[1,1,:,:]),vec_BLOCH_h)
        dHdky[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdky[0,0,:,:]),vec_BLOCH_p)
        dHdky[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdky[1,1,:,:]),vec_BLOCH_h)
        # delete trivial intraband part
        for m in range(DIM):
            for n in range(DIM):
                dHdkx[0,0,m,n] = KD(m,n)*dHdkx[0,0,m,n]         
                dHdkx[1,1,m,n] = KD(m,n)*dHdkx[1,1,m,n]           
                dHdky[0,0,m,n] = KD(m,n)*dHdky[0,0,m,n]         
                dHdky[1,1,m,n] = KD(m,n)*dHdky[1,1,m,n]                                                        
        # transoform back to k-orbital baisis
        dHdkx[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdkx[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdkx[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdkx[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        dHdky[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdky[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdky[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdky[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        ###
        
        # transform to BdG band basis
        dHdkx = dewrap(dHdkx)
        dHdky = dewrap(dHdky)
        
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])   
    return WEIGHT      


def SF_weight_geom_EFF(mu, KPATH, OP, A):
    WEIGHT = np.zeros((2,2),dtype=complex)
    gamma_z = np.zeros((2,2,DIM,DIM))
    gamma_z[0,0,:,:] = np.eye(DIM) 
    gamma_z[1,1,:,:] = -np.eye(DIM) 
    gamma_z = dewrap(gamma_z)
    for k in range(NN**2):
        H = dewrap(set_HK_BdG_EFF(mu,KPATH[k],OP,A))
        ev, vec = np.linalg.eigh(H)
        
        dHdkx = set_dHK_BdGdkx_EFF(KPATH[k],A)
        dHdky = set_dHK_BdGdky_EFF(KPATH[k],A)
        
        ### remove geoemtric (interband) part
        ev_BLOCH_p, vec_BLOCH_p = np.linalg.eigh(H_EFF(KPATH[k],A))                 # eigensystem of particle Hamiltonian (Bloch functions)
        ev_BLOCH_h, vec_BLOCH_h = np.linalg.eigh(-np.conj(H_EFF(-KPATH[k],A)))      # eigensystem of hole Hamiltonian
        
        # transform current into Bloch basis
        dHdkx[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdkx[0,0,:,:]),vec_BLOCH_p)
        dHdkx[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdkx[1,1,:,:]),vec_BLOCH_h)
        dHdky[0,0,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_p)),dHdky[0,0,:,:]),vec_BLOCH_p)
        dHdky[1,1,:,:] = np.dot(np.dot(np.transpose(np.conj(vec_BLOCH_h)),dHdky[1,1,:,:]),vec_BLOCH_h)
        # delete trivial intraband part
        for m in range(DIM):
            dHdkx[0,0,m,m] = 0.       
            dHdkx[1,1,m,m] = 0.         
            dHdky[0,0,m,m] = 0.          
            dHdky[1,1,m,m] = 0.                                                         
        # transoform back to k-orbital baisis
        dHdkx[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdkx[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdkx[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdkx[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        dHdky[0,0,:,:] = np.dot(vec_BLOCH_p,np.dot(dHdky[0,0,:,:],np.transpose(np.conj(vec_BLOCH_p))))
        dHdky[1,1,:,:] = np.dot(vec_BLOCH_h,np.dot(dHdky[1,1,:,:],np.transpose(np.conj(vec_BLOCH_h))))       
        ###
        
        # transform to BdG band basis
        dHdkx = dewrap(dHdkx)
        dHdky = dewrap(dHdky)
        
        dhdkx_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdkx,gamma_z)),vec)
        g_dhdkx = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdkx)),vec)
        dHdkx = np.dot(np.dot(np.transpose(np.conj(vec)),dHdkx),vec)
        
        dhdky_g = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(dHdky,gamma_z)),vec)
        g_dhdky = np.dot(np.dot(np.transpose(np.conj(vec)),np.dot(gamma_z,dHdky)),vec)
        dHdky = np.dot(np.dot(np.transpose(np.conj(vec)),dHdky),vec)
        
        for i in range(2*DIM):
            for j in range(2*DIM):
                if(np.abs(ev[i]-ev[j])<eps):
                    CC = -dfermi(ev[i])
                else:
                    CC = (fermi(ev[j])-fermi(ev[i]))/(ev[i]-ev[j])
                
                WEIGHT[0,0] += CC/NN**2*(dHdkx[i,j]*dHdkx[j,i]-dhdkx_g[i,j]*g_dhdkx[j,i])     
                WEIGHT[0,1] += CC/NN**2*(dHdkx[i,j]*dHdky[j,i]-dhdkx_g[i,j]*g_dhdky[j,i])  
                WEIGHT[1,0] += CC/NN**2*(dHdky[i,j]*dHdkx[j,i]-dhdky_g[i,j]*g_dhdkx[j,i])  
                WEIGHT[1,1] += CC/NN**2*(dHdky[i,j]*dHdky[j,i]-dhdky_g[i,j]*g_dhdky[j,i])   
    return WEIGHT                     

DD_EFF = SF_weight_full_EFF(MU_EFF, MAT_BZ, ORDER_EFF, AA)
DD_TRIV_EFF = SF_weight_trivial_EFF(MU_EFF, MAT_BZ, ORDER_EFF, AA)
DD_GEO_EFF = SF_weight_geom_EFF(MU_EFF, MAT_BZ, ORDER_EFF, AA)
print("weight effective: "+str(DD_EFF))
print("weight effective (trivial): "+str(DD_TRIV_EFF))
print("weight effective (geometric): "+str(DD_GEO_EFF))
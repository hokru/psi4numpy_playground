"""
A reference implementation of density-fitted MP2 from a RHF reference.
with added simple NAF approximiation
http://aip.scitation.org/doi/10.1063/1.4905005

procedure:
J=Qov=(Q|ia)

paper equations: (mind that PSI4's "J" is (Q|ia), paper has (ia|Q))
1. form W=Jt*J
2. diag W
3. select significant eigenvectors of W in matrix N_bar
4. form new J J_bar=J*N_bar
5. calculated MP2 energy with new J=J_bar

References: 
Algorithm modified from Rob Parrish's most excellent Psi4 plugin example
Bottom of the page: http://www.psicode.org/developers.php
"""

__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith", "Dominic A. Sirianni"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4

# Set memory & output
psi4.set_memory('4 GiB')
psi4.core.set_output_file('output.dat', False)

mol = psi4.geometry(""" 
C    1.39410    0.00000   0.00000
C    0.69705   -1.20732   0.00000
C   -0.69705   -1.20732   0.00000
C   -1.39410    0.00000   0.00000
C   -0.69705    1.20732   0.00000
C    0.69705    1.20732   0.00000
H    2.47618    0.00000   0.00000
H    1.23809   -2.14444   0.00000
H   -1.23809   -2.14444   0.00000
H   -2.47618    0.00000   0.00000
H   -1.23809    2.14444   0.00000
H    1.23809    2.14444   0.00000
symmetry c1
""")

# Basis used in mp2 density fitting
#psi4.set_options({'basis': 'cc-pVDZ',
#                  'df_basis_scf': 'cc-pvdz-ri'})

psi4.set_options({'basis': 'minix', 'df_basis_scf': 'def2-SVP-RI'})
check_energy = False
#check_energy = True

print('\nStarting RHF...')
t = time.time()
RHF_E, wfn = psi4.energy('SCF', return_wfn=True)
print('...RHF finished in %.3f seconds:   %16.10f' % (time.time() - t, RHF_E))

# Grab data from Wavfunction clas
ndocc = wfn.nalpha()
nbf = wfn.nso()
nvirt = nbf - ndocc

# Split eigenvectors and eigenvalues into o and v
eps_occ = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_OCC"))
eps_vir = np.asarray(wfn.epsilon_a_subset("AO", "ACTIVE_VIR"))

# Build DF tensors
print('\nBuilding DF ERI tensor Qov...')
t = time.time()
C = wfn.Ca()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", "aug-cc-pvdz")
df = psi4.core.DFTensor(wfn.basisset(), aux, C, ndocc, nvirt)
# Transformed MO DF tensor
Qov = np.asarray(df.Qov())
print('...Qov build in %.3f seconds with a shape of %s, %.3f GB.' \
% (time.time() - t, str(Qov.shape), np.prod(Qov.shape) * 8.e-9))

########## NAF start ###############
#  calculate W Matrix = W=Jt*J
print(' ** NAF  **')
naux = aux.nbf()
print('computing W matrix with dimension naux = %i' % (naux))
# reshape to form J (Q|ia)
J = np.reshape(Qov, (naux, nvirt * ndocc))

# eq. 5 but transposed (Q|Q)=(Q|ia)(ia|Q)
W = np.dot(J, J.T)

print('diagonalization of W ')
e_val, e_vec = np.linalg.eigh(W)
#  construct modified Qov matrix Nbar=(Qbar|Q) from selected eigenvectors of W
print('selecting eigenvectors ')
# ethr threshold is supposed to be 10^-4 ???
ethr = 1e-18   
nskipped = 0
Ntmp = np.zeros((naux, naux))
niter = 0
naux2 = 0

# sanity check if transformations are OK
#for n in range(naux):
#    Nbar[naux2, :] = e_vec[n, :]
#    naux2 += 1

tmp = []
naux2 = 0
for n in range(naux):
#    print(e_val[n])
    if (abs(e_val[n]) >= ethr):
#    if (e_val[n] >= 0.0):
        Ntmp[naux2, :] = e_vec[n, :]
        naux2 += 1

print('new naux = %i  ' % (naux2))
Nbar = Ntmp[0:naux2,0:naux] 

# eq.6 Qov_bar=Jbar=(Qbar|ia)=(Qbar|Q)*(Q|ia)
print('computing new Qov ')
Qov_bar = np.dot(Nbar, J)
print('shape Jbar :', Qov_bar.shape)
Qov = np.zeros((naux2, ndocc, nvirt))
# rebuild 3-index Qov
# numpy solution?
Qov=np.reshape(Qov_bar,(naux2,ndocc,nvirt))  # can i do this?
#for q in range(naux2):
#    for i in range(ndocc):
#        for a in range(nvirt):
#            qia = (q, i, a)
#            idx = np.unravel_index(
#                np.ravel_multi_index(qia, Qov.shape), Qov_bar.shape)
#            #            print(qia,idx)
#            Qov[q, i, a] = Qov_bar[idx]

# print(Qov)
print('shape Qov:', Qov.shape)
########## NAF end  ###############

print('\nComputing MP2 energy...')
t = time.time()
# A smarter algorithm, loop over occupied indices and exploit ERI symmetry

# This part of the denominator is identical for all i,j pairs
vv_denom = -eps_vir.reshape(-1, 1) - eps_vir

MP2corr_OS = 0.0
MP2corr_SS = 0.0
for i in range(ndocc):
    eps_i = eps_occ[i]
    i_Qv = Qov[:, i, :].copy()
    for j in range(i, ndocc):

        eps_j = eps_occ[j]
        j_Qv = Qov[:, j, :]

        # We can either use einsum here
        #        tmp = np.einsum('Qa,Qb->ab', i_Qv, j_Qv)

        # Or a dot product (DGEMM) for speed)
        tmp = np.dot(i_Qv.T, j_Qv)

        # Diagonal elements
        if i == j:
            div = 1.0 / (eps_i + eps_j + vv_denom)
        # Off-diagonal elements
        else:
            div = 2.0 / (eps_i + eps_j + vv_denom)

        # Opposite spin computation
        MP2corr_OS += np.einsum('ab,ab,ab->', tmp, tmp, div)

        # Notice the same-spin compnent has an "exchange" like term associated with it
        MP2corr_SS += np.einsum('ab,ab,ab->', tmp - tmp.T, tmp, div)

print('...finished computing MP2 energy in %.3f seconds.' % (time.time() - t))

MP2corr_E = MP2corr_SS + MP2corr_OS
MP2_E = RHF_E + MP2corr_E

#print('\nMP2 SS correlation energy:         %16.10f' % MP2corr_SS)
#print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)

print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
print('MP2 total energy:                  %16.10f' % MP2_E)

psi4.energy('MP2')
ecorr = psi4.core.get_variable('MP2 CORRELATION ENERGY')
print('reference Ecorr(MP2) = %f ; error = %f ' % (ecorr, ecorr - MP2corr_E))

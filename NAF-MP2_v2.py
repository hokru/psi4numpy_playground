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

psi4.set_options({'basis': 'minix', 'df_basis_mp2': 'cc-pVDZ-RI'})
t = time.time()
# wavefunction object
wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
orb = wfn.basisset()
norb = orb.nbf()

aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", "cc-pVDZ-RI")
naux = aux.nbf()

# Build instance of MintsHelper
mints = psi4.core.MintsHelper(orb)
zero_bas = psi4.core.BasisSet.zero_ao_basis_set()

# Build (P|pq) raw 3-index ERIs, dimension (1, Naux, nbf, nbf)
Ppq = mints.ao_eri(zero_bas, aux, orb, orb)

# Build Coulomb metric but only invert, dimension (1, Naux, 1, Naux)
metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
metric.power(-1.0, 1.e-14)
# metric.power(-0.5, 1.e-14)

# Remove excess dimensions of Ppq, & metric
Ppq = np.squeeze(Ppq)
metric = np.squeeze(metric)

# Build the Qso object
# Qpq = np.einsum('QP,Ppq->Qpq', metric, Ppq)

# paper uses transpose of Ppq, so we adapt for now
Ppq = np.reshape(Ppq, (naux, norb * norb)).T
print("I  = (pq|P) dim:", Ppq.shape)

# cholesky decomp of inverse metric
L = np.linalg.cholesky(metric)
print("L  = cholesky[(P|Q)^1 ]dim:", L.shape)

# Form intermediate W'= I^t*I
Wp = np.dot(Ppq.T, Ppq)
print("W' = (P|P) dim:", Wp.shape)

# form W proper
W = np.dot(np.dot(L.T, Wp), L)
print("W  = (Q|Q) dim:", W.shape)

# from N(bar) from eigenvectors of W
# epsilon threshold is supposed to be in the range of 10^-2 to 10^-4 ?
e_val, e_vec = np.linalg.eigh(W)
# print(e_val)
eps = 1e-1
print(eps)
nskipped = 0
Ntmp = np.zeros((naux, naux))
naux2 = 0
for n in range(naux):
    if (abs(e_val[n]) > eps):
        # print(e_val[n])
        Ntmp[:, naux2] = e_vec[:, n]
        naux2 += 1


print('retaining new naux = %i  of  %i [ %4.1f %% ]' % (naux2,naux,naux2/naux*100.0))
Nbar = Ntmp[0:naux, 0:naux2]
# Nbar=e_vec
print("N^bar  = (Q^bar|Q) dim)",Nbar.shape)

# form N'(bar) = L * N(bar)
Npbar = np.dot(L,Nbar)
print("N'^bar  = (P^bar|Q) dim)",Npbar.shape)

# form J(bar) = I * N'(bar)
Jbar =  np.dot(Ppq,Npbar)
print("J^bar  = (pq|Q) dim)",Npbar.shape)

# HF energy and orbitals
print(' ** RHF  **')
RHF_E, RHFwfn = psi4.energy('SCF', return_wfn=True)

# Grab data from Wavfunction clas
ndocc = RHFwfn.nalpha()
nbf = RHFwfn.nso()
nvirt = nbf - ndocc

# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(RHFwfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]

# Get MO coefficients from SCF wavefunction
C = np.asarray(RHFwfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

# MP2 energy
print(' ** MP2  **')
# transform J(bar) = Qso -> Qov
# expant J(bar) to proper dimensions
Qpq = Jbar.T.reshape(naux2,norb,norb) # can i do this?
# print(Qpq.shape)

# Normal construction
# metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
# metric.power(-0.5, 1.e-14)
# J =  np.dot(Ppq,metric)
# Qpq = J.T.reshape(naux,norb,norb) # can i do this?

# ==> Transform Qpq -> Qmo @ O(N^4) <==
Qmo = np.einsum('pi,Qpq->Qiq', C, Qpq)
Qmo = np.einsum('Qiq,qj->Qij', Qmo, C)

# Get Occupied-Virtual Block
Qmo = Qmo[:, :ndocc, ndocc:]

# print(Qmo)
# MP2 part is correct for correct DF tensor
# df = psi4.core.DFTensor(RHFwfn.basisset(), aux, RHFwfn.Ca(), ndocc, nvirt)
# Qmo = np.asarray(df.Qov())


# # TEST TEST TEST TEST TEST TEST TEST TEST TEST
# Qmo = np.einsum('pi,Qpq->Qiq', C, Qpq)
# Qmo = np.einsum('Qiq,qj->Qij', Qmo, C)
# Qmo = Qmo[:, :ndocc, ndocc:]
# df = psi4.core.DFTensor(RHFwfn.basisset(), aux, RHFwfn.Ca(), ndocc, nvirt)
# Qref = np.asarray(df.Qov())
# print(Qmo-Qref)
# print(Qref)
# sys.exit
# TEST TEST TEST TEST TEST TEST TEST TEST TEST


# ==> Build VV Epsilon Tensor <==
e_vv = e_ab.reshape(-1, 1) + e_ab

mp2_os_corr = 0.0
mp2_ss_corr = 0.0
for i in range(ndocc):
    # Get epsilon_i from e_ij
    e_i = e_ij[i]
    
    # Get 2d array Qa for i from Qov
    i_Qa = Qmo[:, i, :]
    
    for j in range(i, ndocc):
        # Get epsilon_j from e_ij
        e_j = e_ij[j]
        
        # Get 2d array Qb for j from Qov
        j_Qb = Qmo[:, j, :]
        
        # Compute 2d ERI array for fixed i,j from Qa & Qb
        # ij_Iab = np.einsum('Qa,Qb->ab', i_Qa, j_Qb)
        ij_Iab = np.dot(i_Qa.T, j_Qb)

        # Compute energy denominator
        if i == j:
            e_denom = 1.0 / (e_i + e_j - e_vv)
        else:
            e_denom = 2.0 / (e_i + e_j - e_vv)

        # Compute SS & OS MP2 Correlation
        mp2_os_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab, e_denom)
        mp2_ss_corr += np.einsum('ab,ab,ab->', ij_Iab, ij_Iab - ij_Iab.T, e_denom)

# Compute MP2 correlation & total MP2 Energy
mp2_corr = mp2_os_corr + mp2_ss_corr
MP2_E = RHF_E + mp2_corr
print('E(MP2) %f' % (MP2_E))
print('Ecorr(MP2) %f' % (mp2_corr))

print('NAF-DF-MP2 finished in %.3f s' \
% (time.time() - t))

 # ==> Compare to Psi4 <==
e=psi4.energy('mp2')
print('REF(MP2) %f' % (e))
ecorr = psi4.core.get_variable('MP2 CORRELATION ENERGY')
print('reference Ecorr(MP2) = %f ; error = %.3e ' % (ecorr, ecorr - mp2_corr))
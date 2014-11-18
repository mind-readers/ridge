import numpy as np
import logging
from ridge_test import ridge_test, ridge_corr_test, bootstrap_ridge_test
from ridge import ridge, ridge_corr, bootstrap_ridge
from mpi4py import MPI

logging.basicConfig(level=logging.DEBUG)

np.random.seed(0)

# TODO: Add a flag to run code vs run code with solution checking
#       useful for easy profiling

# Create some test data
# Good test values:
#   N=2000, M=30000, TR=10000, TP=2000
N = 200 # features
M = 3000 # response sources (voxels, whatever)
TR = 1000 # regression timepoints
TP = 200 # prediction timepoints

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

realwt = np.random.randn(N, M)
features = np.random.randn(TR+TP, N)
noise = np.random.randn(TR+TP, M)


snrs = np.linspace(0, 0.2, M)
realresponses = np.dot(features, realwt) # shape (TR+TP, M)
responses = (realresponses * snrs) + noise

Rresp = responses[:TR]
Presp = responses[TR:]
Rstim = features[:TR]
Pstim = features[TR:]

# Run bootstrap ridge
wt_test, corr_test, valphas_test, bscorrs_test, valinds_test = bootstrap_ridge_test(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=12,
                                                      chunklen=10, nchunks=15, test_bootstrap=True)

wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=12,
                                                      chunklen=10, nchunks=15, test_bootstrap=True)

#for i in range(wt.shape[0]):
#    for j in range(wt.shape[1]):
#        if wt[i][j] != wt_test[i][j]:
#            print "fail in wt: " + str(wt[i][j]) + "!=" + str(wt_test[i][j]) + " at " + str(i) + ", " + str(j)
#            exit(1)

if not np.allclose(wt, wt_test):
    print("Fail in wt")
    exit(1)

#for i in range(corr.shape[0]):
#    if corr[i] != corr_test[i]:
#        print "fail in corr"
#        exit(1)

if not np.allclose(corr, corr_test):
    print("Fail in corr")
    exit(1)

#for i in range(valphas.shape[0]):
#    if valphas[i] != valphas_test[i]:
#        print "fail in valphas"
#        exit(1)

if not np.allclose(valphas, valphas_test):
    print("Fail in valphas")
    exit(1)

#for i in range(bscorrs.shape[0]):
#    for j in range(bscorrs.shape[1]):
#        for k in range(bscorrs.shape[2]):
#            if bscorrs[i][j][k] != bscorrs_test[i][j][k]:
#                print "fail in bootstrap_corrs"
#                exit(1)

if not np.allclose(bscorrs, bscorrs_test):
    print("Fail in bscorrs")
    exit(1)

#for i in range(valinds.shape[0]):
#    for j in range(valinds.shape[1]):
#        if valinds[i][j] != valinds_test[i][j]:
#            print "fail in valinds"
#            exit(1)

if not np.allclose(valinds, valinds_test):
    print("Fail valinds")
    exit(1)


#Corr should increase quickly across "voxels". Last corr should be large (>0.9-ish).
# wt should be very similar to realwt for last few voxels.

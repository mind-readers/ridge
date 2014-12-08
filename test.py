import numpy as np
import logging
from ridge_test import ridge_test, ridge_corr_test, bootstrap_ridge_test
from ridge import ridge, ridge_corr, bootstrap_ridge
from mpi4py import MPI
import sys
import time

logging.basicConfig(level=logging.DEBUG)

np.random.seed(0)

# Do some manual arg parsing, kinda crufty
if len(sys.argv) == 2:
    arg = sys.argv[1]
    if arg not in ["--benchmark", "--test-correct"]:
        print("Bad argument, exitting")
else:
    print("No argument for what to test/benchmark")

# Create some test data
# Good test values:
#   N=2000, M=30000, TR=10000, TP=2000
# Use small data size for testing correctness
if arg == "--test-correct":
    N = 400
    M = 3000
    TR = 1000
    TP = 200
else:
    # N and M are 1/2 of what is used in practice
    N = 30000 # features
    M = 60000 # response sources (voxels, whatever)
    TR = 1000 # regression timepoints
    TP = 200 # prediction timepoints

# Benchmark sizes:
# Fixed values
# M = always 30,000
# TR = always 1000
# TP = always 200
# Varying values
# N = 30000, 15000, 7500, 4000, 2000, 1000, 500

# Reasonable realistic values:
# N  = 30000
# M  = 60000
# TR = 5000
# TP = 1000

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
Pstim = np.copy(Pstim, order='F')

if arg == "--benchmark" and rank == 0:
    f = open("benchmark.log", "a")
    t0 = time.time()

# Run the optimized version of the ridge code
wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=15,
                                                      chunklen=10, nchunks=15, test_bootstrap=True)

if arg == "--benchmark" and rank == 0:
    t1 = time.time()
    total_runtime = t1 - t0
    f.write("Runtime: N=%d, M=%d, TR=%d, TP=%d, Time=%d\n" % (N, M, TR, TP, total_runtime))
    f.close()



# Run the original ridge code, if we are correctness testing only
if arg == "--test-correct":
    print("Running original ridge code...")
    wt_test, corr_test, valphas_test, bscorrs_test, valinds_test = bootstrap_ridge_test(Rstim, Rresp, Pstim, Presp,
                                                          alphas=np.logspace(-2, 2, 20),
                                                          nboots=15,
                                                          chunklen=10, nchunks=15, test_bootstrap=True)
    
    print("Comparing answers between optimized ridge and original ridge")
    
    if not np.allclose(wt, wt_test):
        print("Fail in wt")
        exit(1)
    
    if not np.allclose(corr, corr_test):
        print("Fail in corr")
        exit(1)
    
    if not np.allclose(valphas, valphas_test):
        print("Fail in valphas")
        exit(1)
    
    if not np.allclose(bscorrs, bscorrs_test):
        print("Fail in bscorrs")
        exit(1)
    
    if not np.allclose(valinds, valinds_test):
        print("Fail valinds")
        exit(1)
    


# Below is some legacy test code we keep around for easy/quick reference
#
#for i in range(wt.shape[0]):
#    for j in range(wt.shape[1]):
#        if wt[i][j] != wt_test[i][j]:
#            print "fail in wt: " + str(wt[i][j]) + "!=" + str(wt_test[i][j]) + " at " + str(i) + ", " + str(j)
#            exit(1)

#for i in range(corr.shape[0]):
#    if corr[i] != corr_test[i]:
#        print "fail in corr"
#        exit(1)

#for i in range(valphas.shape[0]):
#    if valphas[i] != valphas_test[i]:
#        print "fail in valphas"
#        exit(1)

#for i in range(bscorrs.shape[0]):
#    for j in range(bscorrs.shape[1]):
#        for k in range(bscorrs.shape[2]):
#            if bscorrs[i][j][k] != bscorrs_test[i][j][k]:
#                print "fail in bootstrap_corrs"
#                exit(1)

#for i in range(valinds.shape[0]):
#    for j in range(valinds.shape[1]):
#        if valinds[i][j] != valinds_test[i][j]:
#            print "fail in valinds"
#            exit(1)



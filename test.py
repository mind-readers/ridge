import numpy as np
import logging
from ridge_test import ridge_test, ridge_corr_test, bootstrap_ridge_test
from ridge import ridge, ridge_corr, bootstrap_ridge

logging.basicConfig(level=logging.DEBUG)

# Create some test data
N = 200 # features
M = 1000 # response sources (voxels, whatever)
TR = 1000 # regression timepoints
TP = 200 # prediction timepoints

snrs = np.linspace(0, 0.2, M)
realwt = np.random.randn(N, M)
features = np.random.randn(TR+TP, N)
realresponses = np.dot(features, realwt) # shape (TR+TP, M)
noise = np.random.randn(TR+TP, M)
responses = (realresponses * snrs) + noise

Rresp = responses[:TR]
Presp = responses[TR:]
Rstim = features[:TR]
Pstim = features[TR:]

# Run bootstrap ridge
wt_test, corr_test, valphas_test, bscorrs_test, valinds_test = bootstrap_ridge_test(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=5,
                                                      chunklen=10, nchunks=15, test_bootstrap=True)

wt, corr, valphas, bscorrs, valinds = bootstrap_ridge(Rstim, Rresp, Pstim, Presp,
                                                      alphas=np.logspace(-2, 2, 20),
                                                      nboots=5,
                                                      chunklen=10, nchunks=15, test_bootstrap=True)

for wt_cur in wt_test:
    if wt_cur not in wt:
        print "fail in wt"
        return

for corr_cur in corr_test:
    if corr_cur not in corr:
        print "fail in corr"
        return

for valphas_cur in valphas_test:
    if valphas_cur not in valphas:
        print "fail in valphas"
        return

for bscorrs_cur in bscorrs_test:
    if bscorrs_cur not in bscorrs:
        print "fail in bscorrs"
        return

for valinds_cur in valinds_test:
    if valinds_cur not in valinds:
        print "fail in valinds"
        return


# Corr should increase quickly across "voxels". Last corr should be large (>0.9-ish).
# wt should be very similar to realwt for last few voxels.

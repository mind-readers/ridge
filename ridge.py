#import scipy
import numpy as np
import logging
from utils import mult_diag, counter
import random
import itertools as itools
from mpi4py import MPI

import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import scikits.cuda.linalg as linalg
import scikits.cuda.misc as misc
linalg.init()

zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score function

ridge_logger = logging.getLogger("ridge_corr")

def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        # TODO determine if this should be a GPU op
        # stim is TRxN (~1000x200 or 5000x15000)
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError, e:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    # EXAMPLE FOR RUNNING GPU LINALG OPERATIONS
    # Export data to GPU
    # U_gpu = gpuarray.to_gpu(U)
    # Do a transpose op on GPU
    # UT_gpu = linalg.transpose(U_gpu)
    # Export more data to GPU
    # resp_gpu = gpuarray.to_gpu(np.nan_to_num(resp))
    # Run a dot product
    # UR_gpu = linalg.dot(UT_gpu, resp_gpu)
    # Fetch data from the GPU
    # UR = UR_gpu.get()
    # The above GPU code can replace the following line:
    UR = np.dot(U.T, np.nan_to_num(resp)) 
    # TODO determine if this should be a GPU op
    # U is output from SVD, I think TRxTR (~1000x1000 or 5000x5000)
    # resp is TRxM (~1000x3000 or 5000x30000)
    
    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, float):
        alpha = np.ones(resp.shape[1]) * alpha
    
    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Use MPI to compute most of the columns of wt
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    MAX_MPI_SIZE = 600000 # Assuming mpi breaks after a certain size

    print("Starting weight computation with %d workers and %d alphas" % (size, len(nalphas)))
    print("Num unique alphas: %d" % (len(np.unique(nalphas)),))

    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]), order='F') # Make wt column major
    # Precompute all selvox values so each mpi job can operate on them independently
    all_selvox = [np.nonzero(nalphas==ualphas[i])[0] for i in range(len(ualphas))]
    num_mpi_rounds = len(ualphas) / size
    remainder_rounds = len(ualphas) % size
    ualphas_rem = num_mpi_rounds*size

    for c in range(num_mpi_rounds):
        # The length of the selvox is how many columns awt will have.
        # We want to find the max length of all selvox in this round so that mpi
        # can pass a consistent array size
        maxlen_selvox = 0
        for sx in range(size):
            len_selvox = all_selvox[c*size+sx].shape[0]
            if len_selvox > maxlen_selvox:
                maxlen_selvox = len_selvox
        ua = ualphas[c*size+rank]
        selvox = all_selvox[c*size+rank] # list of indices equal to ua
        awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        padded_awt = np.zeros((wt.shape[0], maxlen_selvox), order='F')
        # Stick the awt inside padded_awt aligned from the top left corner
        padded_awt[:,:selvox.shape[0]] = awt
        recv_awt = np.empty((wt.shape[0], maxlen_selvox*size), order='F')
        for k in range(padded_awt.shape[1]):
            # Gather a column from each worker
            recv_awt = np.empty((wt.shape[0], size), order='F')
            comm.Allgather(padded_awt[:,k], recv_awt)
            # Put the gathered columns in the right places in recv_awt
            for i in range(size):
                # If this unique alpha has a kth entry, then fill in the column
                if len(all_selvox[c*size+i]) > k:
                    wt[:,all_selvox[c*size+i][k]] = recv_awt[:,i]

    # Compute the remainder rounds that don't divide evenly into the mpi size
    # Similar operations to before
    if remainder_rounds > 0:
        maxlen_selvox = 0
        for sx in range(remainder_rounds):
            len_selvox = all_selvox[ualphas_rem+sx].shape[0]
            if len_selvox > maxlen_selvox:
                maxlen_selvox = len_selvox
        if rank < remainder_rounds:
            ua = ualphas[ualphas_rem+rank]
            selvox = all_selvox[ualphas_rem+rank] # list of indices equal to ua
            awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
            padded_awt = np.empty((wt.shape[0], maxlen_selvox), order='F')
            padded_awt[:,:selvox.shape[0]] = awt
        else:
            # this is just to create an array of the proper size to appease mpi
            padded_awt = np.empty((wt.shape[0], maxlen_selvox), order='F')
        recv_awt = np.empty((wt.shape[0], maxlen_selvox*size), order='F')
        for k in range(padded_awt.shape[1]):
            # Gather a column from each worker
            recv_awt = np.empty((wt.shape[0], size), order='F')
            comm.Allgather(padded_awt[:,k], recv_awt)
            # Put the gathered columns in the right places in recv_awt
            for i in range(size):
                # If this unique alpha has a kth entry, then fill in the column
                if i < remainder_rounds and len(all_selvox[ualphas_rem+i]) > k:
                    wt[:,all_selvox[ualphas_rem+i][k]] = recv_awt[:,i]
        #comm.Allgather(padded_awt, recv_awt)
        #recv_awt = np.hsplit(recv_awt, size)
        #for i in range(size):
            #if i < remainder_rounds: # This is to pick out the real arrays
                #wt[:,all_selvox[ualphas_rem+i]] = recv_awt[i][:,:all_selvox[ualphas_rem+i].shape[0]]


    ## Compute weights for each alpha
    #ualphas = np.unique(nalphas)
    #wt = np.zeros((stim.shape[1], resp.shape[1]), order='F') # Make wt column major
    #for ua in ualphas:
    #    selvox = np.nonzero(nalphas==ua)[0] # list of indices equal to ua
    #    # TODO determine if this should be a GPU op
    #    # Vh is output from SVD, i think NxN (~200x200 or 15000x15000)
    #    # TODO determine how reduce works
    #    awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
    #    wt[:,selvox] = awt
    return wt
    

def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError, e:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S>singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    # TODO determine if this should be a GPU 
    # U is svd output. I think TRxTR (~1000x1000 or 5000x5000)
    # Rresp is TRxM (~1000x3000 or 5000x30000)
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    # TODO determine if this should be a GPU op
    # Pstim is TPxN (~200x200 or 1000x15000)
    # Vh is output from SVD, I think NxN (~200x200 or 15000x15000)
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    Prespvar = Presp.var(0)
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S/(S**2+na**2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        # TODO determine if this should be a GPU op
        # mult_diag is diagonal matrix. 
        # UR is TRxM (~1000x3000 or 5000x30000)
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp*zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp-pred).var(0)
            Rcorr = np.clip(1-(resvar/Prespvar), 0, 1)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        #log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        #log_msg = log_template % (a,
        #                          np.mean(Rcorr),
        #                          np.max(Rcorr),
        #                          corrmin,
        #                          (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        #logger.info(log_msg)
    
    return Rcorrs


def bootstrap_ridge(Rstim, Rresp, Pstim, Presp, alphas, nboots, chunklen, nchunks,
                    corrmin=0.2, joined=None, singcutoff=1e-10, normalpha=False, single_alpha=False,
                    use_corr=True, logger=ridge_logger, test_bootstrap=False):
    """Uses ridge regression with a bootstrapped held-out set to get optimal alpha values for each response.
    [nchunks] random chunks of length [chunklen] will be taken from [Rstim] and [Rresp] for each regression
    run.  [nboots] total regression runs will be performed.  The best alpha value for each response will be
    averaged across the bootstraps to estimate the best alpha for that response.
    
    If [joined] is given, it should be a list of lists where the STRFs for all the voxels in each sublist 
    will be given the same regularization parameter (the one that is the best on average).
    
    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M different responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M different responses. Each response should be Z-scored across
        time.
    alphas : list or array_like, shape (A,)
        Ridge parameters that will be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    nboots : int
        The number of bootstrap samples to run. 15 to 30 works well.
    chunklen : int
        On each sample, the training data is broken into chunks of this length. This should be a few times 
        longer than your delay/STRF. e.g. for a STRF with 3 delays, I use chunks of length 10.
    nchunks : int
        The number of training chunks held out to test ridge parameters for each bootstrap sample. The product
        of nchunks and chunklen is the total number of training samples held out for each sample, and this 
        product should be about 20 percent of the total length of the training data.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested for each bootstrap sample, the number of 
        responses with correlation greater than this value will be printed. For long-running regressions this
        can give a rough sense of how well the model works before it's done.
    joined : None or list of array_like indices
        If you want the STRFs for two (or more) responses to be directly comparable, you need to ensure that
        the regularization parameter that they use is the same. To do that, supply a list of the response sets
        that should use the same ridge parameter here. For example, if you have four responses, joined could
        be [np.array([0,1]), np.array([2,3])], in which case responses 0 and 1 will use the same ridge parameter
        (which will be parameter that is best on average for those two), and likewise for responses 2 and 3.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    normalpha : boolean
        Whether ridge parameters (alphas) should be normalized by the largest singular value (LSV)
        norm of Rstim. Good for rigorously comparing models with different numbers of parameters.
    single_alpha : boolean
        Whether to use a single alpha for all responses. Good for identification/decoding.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.
    
    Returns
    -------
    wt : array_like, shape (N, M)
        Regression weights for N features and M responses.
    corrs : array_like, shape (M,)
        Validation set correlations. Predicted responses for the validation set are obtained using the regression
        weights: pred = np.dot(Pstim, wt), and then the correlation between each predicted response and each 
        column in Presp is found.
    alphas : array_like, shape (M,)
        The regularization coefficient (alpha) selected for each voxel using bootstrap cross-validation.
    bootstrap_corrs : array_like, shape (A, M, B)
        Correlation between predicted and actual responses on randomly held out portions of the training set,
        for each of A alphas, M voxels, and B bootstrap samples.
    valinds : array_like, shape (TH, B)
        The indices of the training data that were used as "validation" for each bootstrap sample.
    """
    nresp, nvox = Rresp.shape
    bestalphas = np.zeros((nboots, nvox))  # Will hold the best alphas for each voxel

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Keep it super simple by enforcing that the num of
    # bootstraps be divisible by the number of workers;
    # else, the number of bootstraps will be truncated
    local_boots = nboots / size
    if (nboots/size)*size != nboots:
        logger.info("Number of workers do not cleanly divide requested bootstrap count. Doing %d bootstraps total instead" % (local_boots*size,))
    if test_bootstrap:
        k = rank
    else:
        k = None

    Rcmats = []
    valinds = [] # Will hold the indices into the validation data for each bootstrap
    for i in range(local_boots):
        if k < nboots:
            logger.info("Rank " + str(rank) + " running bootstrap " + str(i+1) +
                    "/"+ str(local_boots) + " with seed " + str(k))
            if test_bootstrap:
                random.seed(k)
            logger.info("Selecting held-out test set..")
            allinds = range(nresp)
            indchunks = zip(*[iter(allinds)]*chunklen)
            random.shuffle(indchunks)
            logger.info(str(indchunks[0:3]))
            heldinds = list(itools.chain(*indchunks[:nchunks]))
            notheldinds = list(set(allinds)-set(heldinds))
            
            RRstim = Rstim[notheldinds,:]
            PRstim = Rstim[heldinds,:]
            RRresp = Rresp[notheldinds,:]
            PRresp = Rresp[heldinds,:]
            
            # Run ridge regression using this test set
            Rcmat = ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                               corrmin=corrmin, singcutoff=singcutoff,
                               normalpha=normalpha, use_corr=use_corr,
                               logger=logger)
        else:
            Rcmat = None
            heldinds = None
            
        Rcmat = np.array(Rcmat)
        # Allocate an empty numpy array to hold MPI collected data
        recv_Rcmats = np.empty((size*len(alphas), nvox), dtype=np.float64)
        comm.barrier()
        comm.Allgather(Rcmat, recv_Rcmats)
        # Split recv'd data into 'size' separate arrays (from each worker)
        Rcmats += np.split(recv_Rcmats, size)
        comm.barrier()
        valinds += comm.allgather(heldinds)
        comm.barrier()

        if test_bootstrap:
            k += size


#    for local_bootstrap_result in global_Rcmats:
#        Rcmats += local_bootstrap_result
#    for local_valinds_result in global_valinds:
#        valinds += local_valinds_result
    valinds = [ x for x in valinds if x != None ]
    valinds = np.array(valinds)

    
    # Find best alphas
    if nboots>0:
        Rcmats = [ x for x in Rcmats if x != None ]
        allRcorrs = np.dstack(Rcmats)
    else:
        allRcorrs = None
    
    if not single_alpha:
        if nboots==0:
            raise ValueError("You must run at least one cross-validation step to assign "
                             "different alphas to each response.")
        
        logger.info("Finding best alpha for each voxel..")
        if joined is None:
            # Find best alpha for each voxel
            meanbootcorrs = allRcorrs.mean(2)
            bestalphainds = np.argmax(meanbootcorrs, 0)
            valphas = alphas[bestalphainds]
        else:
            # Find best alpha for each group of voxels
            valphas = np.zeros((nvox,))
            for jl in joined:
                # Mean across voxels in the set, then mean across bootstraps
                jcorrs = allRcorrs[:,jl,:].mean(1).mean(1)
                bestalpha = np.argmax(jcorrs)
                valphas[jl] = alphas[bestalpha]
    else:
        logger.info("Finding single best alpha..")
        if nboots==0:
            if len(alphas)==1:
                bestalphaind = 0
                bestalpha = alphas[0]
            else:
                raise ValueError("You must run at least one cross-validation step "
                                 "to choose best overall alpha, or only supply one"
                                 "possible alpha value.")
        else:
            meanbootcorr = allRcorrs.mean(2).mean(1)
            bestalphaind = np.argmax(meanbootcorr)
            bestalpha = alphas[bestalphaind]
        
        valphas = np.array([bestalpha]*nvox)
        logger.info("Best alpha = %0.3f"%bestalpha)

    # Find weights
    logger.info("Computing weights for each response using entire training set..")
    wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)
    #comm.Barrier()
    #logger.info("Finished computing results of ridge()...")
    #if rank == 0:
        #logger.info("Computing weights for each response using entire training set..")
        #wt = ridge(Rstim, Rresp, valphas, singcutoff=singcutoff, normalpha=normalpha)
        #for nd in range(1, size):
            #comm.Isend(wt, dest=nd)
        #logger.info("Broadcasting results of ridge()...")
    #else:
        #logger.info("Not Computing weights..")
        #wt = np.empty((Rstim.shape[1], Rresp.shape[1]), order='F')
        #logger.info(str(rank) + " Ready to recieve results of ridge()...")
        #comm.Recv(wt, source=0)
        #logger.info(str(rank) + " Recieved results of ridge()...")
    #comm.barrier()
    #wt = comm.Bcast(wt, root=0)

    # Predict responses on prediction set
    logger.info("Predicting responses for predictions set..")
    pred = np.dot(Pstim, wt)

    # Find prediction correlations
    nnpred = np.nan_to_num(pred)
    corrs = np.nan_to_num(np.array([np.corrcoef(Presp[:,ii], nnpred[:,ii].ravel())[0,1]
                                    for ii in range(Presp.shape[1])]))

    return wt, corrs, valphas, allRcorrs, valinds

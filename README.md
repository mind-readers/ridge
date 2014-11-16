ridge
=====

*This forked repo is for optimizing some portion (or all) of this ridge regression procedure, taking advantage of parallel architecture strategies developed
in UC Berkeley's Computer Science 194 course on Parallel Software Engineering*

This is an implementation of [ridge regression](http://en.wikipedia.org/wiki/Tikhonov_regularization) (aka L2-regularized regression or Tikhonov regression) that takes advantage of some linear algebra tricks to do very efficient cross validation. This method is particularly useful when the number of models that you are trying to fit simultaneously is very large (thousands to tens of thousands), the number of features is very large (thousands), and the number of data points for each model is very large (thousands). 

Even for a pretty modest ridge regression problem (20k models, 2k features, 1k data points), this method takes 9x less time to fit (80 seconds vs. 718 seconds) versus scikit-learn's cross validated ridge regression (not [GCV](http://en.wikipedia.org/wiki/Tikhonov_regularization#Determination_of_the_Tikhonov_factor)).

This code was developed mainly for building voxel-wise models of fMRI data, where we often want to fit tens of thousands of models simultaneously (1 for each voxel) with thousands of features and thousands of data points.

### Ridge regression (in general)
The goal of ridge regression is to find a linear transformation of your feature matrix, `X`, that best approximates your observed data, `Y`. (Because this code was developed for use in fMRI analyses, we use the terms "stimuli" to refer to `X` and "responses" to refer to `Y`.) The linear transformation takes the form of a weight matrix, `B`, such that `X * B ~ Y`.

In ridge regression, `B` is obtained by taking the ridge pseudoinverse of `X` and multiplying it by `Y`. To get the ridge psuedoinverse we first take the singular value decomposition (SVD) of `X`: `X = U * S * V.T`. For a normal pseudoinverse we would just invert the singular values (forming the inverse matrix `D` by taking `1/S` for each entry in `S`), but for a ridge pseudoinverse we regularize the inverse using a ridge penalty, `a`. Thus we use `D_i = S_i / (S_i^2 + a^2)`. This fixes problems with very small singular values, which would get very large in the inverse and mess things up.

The key issue for doing ridge regression is choosing the right `a`. For real-world data (which is autocorrelated and messy), this is usually done by testing many different possible values of `a` using cross validation. In cross validation the regression dataset is broken up into two parts, a training set and a test set. A separate weight matrix, `B`, is obtained for each value of `a` using the training set, and then that `B` is used to predict the test set. This process is usually repeated a few times for a few different selections of training and test set. Then the best `a` is selected based on how well each `a` could be used to predict the test set. Unfortunately, this process can be extremeley time consuming.

### Optimized cross validation
The innovation of this ridge library is a better way to do cross validation which is much more efficient than a naive implementation. The key insight is that it is not necessary to actually compute `B` for every `a`. It is actually much more efficient to directly compute predictions than it is to compute `B`. This is because the computational complexity of matrix multiplication is not associative (although the outcome is associative).

In the traditional approach, predictions are obtained by multiplying the estimated weights, `B` by the features, `X`. If the SVD of `X` is given as `U * S * V.T` and the ridge diagonal matrix is `D`, then the predictions for a new dataset, (`X_new`, `Y_new`), are computed like this: `Y_new_predicted = X_new * B = X_new * (V * D * U.T * Y)`. Instead, we re-arrange the parentheses and compute predictions like this: `Y_new_predicted = (X_new * V) * D * (U.T * Y)`.

Why does this help? We can figure that out by determining the number of actual multiplications that are done in each case. Let's say that `X` has N features and TR data points, `Y` has M responses and TR data points, and `X_new` and `Y_new` have the same number of features and responses but TP data points. Thus the number of multiplications necessary for the traditional method of finding `Y_new_predicted` is `N^3 + N^2*TR + N*TR*M + TP*N*M`. This cost can be partially offset by caching `U.T * Y`, which reduces the cost by `N*TR*M`. Thus the total cost is about `N^3 + N^2*TR + TP*N*M`.

The number of multiplications necessary for the optimized approach is `TP*N^2 + N^2*TP + N*TR*M + TP*N*M`. This cost can be further offset by caching `X * V` (getting rid of the `TP*N^2`) and `U.T * Y` (getting rid of the `N*TR*M`). Thus the total optimized cost is about `N^2*TP + TP*N*M`. Effectively, this totally gets rid of the `N^3` cost (which can be very large if the number of features is large) and reduces the other cost by a factor of `TP/TR` (TP is usually much smaller than TR).

Dependencies
========
- PyCUDA
- Scikits.cuda
- Mako

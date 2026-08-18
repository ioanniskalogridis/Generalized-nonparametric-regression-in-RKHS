library(Rcpp)
library(RcppArmadillo)

# Import the C++ functions
sourceCpp("rkhs_quan.cpp")

# fit_rkhs is the wrapper for the C++ functions performing LS and quantile kernel regression
# X is a matrix of predictors
# y is the response vector
# loss is the loss function to be used for the estimation, currently only ls and 
# quantile are supported
# kernel refers to the reproducing kernel: currently only gaussian, matern and
# tensor (product matern) are supported
# The kernels can be used for univariate as well as for multivariate data
# tau is the quantile to be estimated
# s is the smoothness parameter for the matern kernel
# To generate the sobolev space of order m on R^d use s = m+d/2
# ls is the bandwidth, by default equal 1
# lambda_grid are the candidate lambdas to be considered for OCV

fit_rkhs <- function(X, y,
                     loss = c("ls", "quantile", "huber"),
                     kernel = c("gaussian", "matern", "tensor"),
                     tau = 0.5,
                     s = 1.5,
                     ls = 1.0,
                     lambda_grid = 10^seq(-7, -1, length.out = 50)
                     ) {
  
  loss <- match.arg(loss)
  kernel <- match.arg(kernel)
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  
  K <- kernel_mat(X, X, kernel, ls, s)
  scores <- numeric(length(lambda_grid))
  
  for (i in seq_along(lambda_grid)) {
    lam <- lambda_grid[i]
    
    if (loss == "ls") {
      fit <- rkhs_ls(K, y, lam)
      scores[i] <- rkhs_ls_ocv(K, y, lam)
    } else if (loss == "quantile") {
      fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lam)
      alpha <- as.numeric(fit$alpha)
      w <- as.numeric(fit$weights)
      scores[i] <- rkhs_quantile_gcv(K, y, alpha, w, lam)
    } else {   # huber
      fit <- rkhs_huber_irls(K, y, lambda = lam)
      scores[i] <- rkhs_huber_gcv(K, y, fit$alpha, fit$weights, lam)
    }
  }
  
  lambda <- lambda_grid[which.min(scores)]

  if (loss == "ls") {
    fit <- rkhs_ls(K, y, lambda)
    result <- list(
      alpha = fit$alpha,
      fitted = fit$fitted,
      lambda = lambda,
      kernel = kernel,
      loss = "ls",
      s = s,
      ls = ls,
      X_train = X
    )
  } else if (loss == "quantile") {
    fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lambda)
    result <- list(
      alpha = fit$alpha,
      fitted = fit$fitted,
      weights = fit$weights,
      lambda = lambda,
      kernel = kernel,
      loss = "quantile",
      tau = tau,
      s = s,
      ls = ls,
      X_train = X
    )
  } else {
    fit <- rkhs_huber_irls(K, y, lambda = lambda, delta = 1.0)
    
    result <- list(
      alpha = fit$alpha,
      fitted = fit$fitted,
      weights = fit$weights,
      lambda = lambda,
      kernel = kernel,
      loss = "huber",
      delta = 1.0,
      s = s,
      ls = ls,
      X_train = X
    )
  }
  
  result$predictor <- function(X_new) {
    X_new <- as.matrix(X_new)
    Knew <- kernel_mat(result$X_train, X_new, kernel, result$ls, result$s)
    as.vector(t(Knew) %*% result$alpha)
  }
  
  result
}

predict_rkhs <- function(fit, X_train, X_new) {
  fit$predictor(X_new)
}

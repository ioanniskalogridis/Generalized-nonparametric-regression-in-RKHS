setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)
sourceCpp("rkhs_quan.cpp")

# ============================================================
# FINAL CLEAN RKHS WRAPPER
# ============================================================
fit_rkhs <- function(X, y,
                     loss = c("ls", "quantile"),
                     kernel = c("gaussian", "matern", "tensor"),
                     tau = 0.5,
                     s = 2.5,
                     ls = 1.0,
                     lambda_grid = 10^seq(-7, -1, length.out = 30),
                     cv = TRUE,
                     verbose = FALSE) {
  
  loss <- match.arg(loss)
  kernel <- match.arg(kernel)
  
  X_train <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X_train)
  
  # Kernel once
  K <- kernel_mat(X_train, X_train, kernel, ls, s)
  
  scores <- numeric(length(lambda_grid))
  
  for (i in seq_along(lambda_grid)) {
    lam <- lambda_grid[i]
    
    if (loss == "ls") {
      fit <- rkhs_ls(K, y, lam)
      scores[i] <- rkhs_ls_ocv(K, y, lam)
    } else {
      fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lam)
      alpha <- as.numeric(fit$alpha)
      w <- as.numeric(fit$weights)
      scores[i] <- rkhs_quantile_gcv(K, y, alpha, w, lam)
    }
  }
  
  lambda <- lambda_grid[which.min(scores)]
  if (verbose) cat("Selected lambda:", format(lambda, scientific = TRUE), "\n")
  
  # Final fit
  if (loss == "ls") {
    fit <- rkhs_ls(K, y, lambda)
    result <- list(
      alpha = as.numeric(fit$alpha),
      fitted = as.numeric(fit$fitted),
      lambda = lambda,
      kernel = kernel,
      loss = "ls",
      s = s,
      ls = ls,
      X_train = X_train
    )
  } else {
    fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lambda)
    result <- list(
      alpha = as.numeric(fit$alpha),
      fitted = as.numeric(fit$fitted),
      weights = as.numeric(fit$weights),
      lambda = lambda,
      kernel = kernel,
      loss = "quantile",
      tau = tau,
      s = s,
      ls = ls,
      X_train = X_train
    )
  }
  
  # Predictor
  result$predictor <- function(X_new) {
    X_new <- as.matrix(X_new)
    Knew <- kernel_mat(result$X_train, X_new, kernel, result$ls, result$s)
    as.vector(t(Knew) %*% result$alpha)
  }
  
  result
}

# Simple alias
predict_rkhs <- function(fit, X_new) {
  fit$predictor(X_new)
}
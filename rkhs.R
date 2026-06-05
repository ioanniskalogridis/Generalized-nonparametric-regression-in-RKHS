setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)
sourceCpp("rkhs_quan.cpp")

# ============================================================
# OPTIMIZED RKHS FIT WRAPPER
# ============================================================
fit_rkhs <- function(X, y,
                     loss = c("ls", "quantile"),
                     kernel = c("gaussian", "matern", "tensor"),
                     tau = 0.5,
                     s = 2.5,
                     ls = 1.0,
                     lambda_grid = 10^seq(-7, -1, length.out = 30),  # reduced default
                     cv = TRUE,
                     verbose = FALSE) {
  
  loss <- match.arg(loss)
  kernel <- match.arg(kernel)
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  
  # ============================================================
  # COMPUTE KERNEL MATRIX ONLY ONCE (Biggest speedup)
  # ============================================================
  K <- kernel_mat(X, X, kernel, ls, s)
  
  # ============================================================
  # CV SCORE STORAGE
  # ============================================================
  scores <- numeric(length(lambda_grid))
  
  # ============================================================
  # MODEL SELECTION LOOP (now much faster)
  # ============================================================
  for (i in seq_along(lambda_grid)) {
    lam <- lambda_grid[i]
    
    if (loss == "ls") {
      fit <- rkhs_ls(K, y, lam)
      scores[i] <- rkhs_ls_ocv(K, y, lam)
    } else {  # quantile
      fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lam)
      alpha <- as.numeric(fit$alpha)
      w <- as.numeric(fit$weights)
      scores[i] <- rkhs_quantile_gcv(K, y, alpha, w, lam)
    }
  }
  
  # Select best lambda
  best_idx <- which.min(scores)
  lambda <- lambda_grid[best_idx]
  
  if (verbose) {
    cat("Selected lambda:", format(lambda, scientific = TRUE), "\n")
  }
  
  # ============================================================
  # FINAL FIT
  # ============================================================
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
      X_train = X
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
      X_train = X
    )
  }
  
  # ============================================================
  # Predictor (cached X_train)
  # ============================================================
  result$predictor <- function(X_new) {
    X_new <- as.matrix(X_new)
    Knew <- kernel_mat(result$X_train, X_new, kernel, ls, s)
    as.vector(t(Knew) %*% result$alpha)
  }
  
  result
}

predict_rkhs <- function(fit, X_train, X_new, ...) {
  Knew <- kernel_mat(X_train, X_new, fit$kernel, ls = fit$ls, s = fit$s)
  as.vector(t(Knew) %*% fit$alpha)
}
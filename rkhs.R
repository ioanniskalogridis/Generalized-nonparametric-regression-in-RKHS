setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_quan.cpp")

# ============================================================
# Main RKHS fit
# ============================================================
fit_rkhs <- function(X, y,
                     loss = c("ls", "quantile"),
                     kernel = c("gaussian", "matern", "spherical"),
                     tau = 0.5,
                     m = 2.5,
                     ls = 1.0,
                     lambda_grid = 10^seq(-8, -1, length.out = 50),
                     cv = TRUE,
                     verbose = FALSE) {
  
  loss   <- match.arg(loss)
  kernel <- match.arg(kernel)
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  
  # ------------------------------------------------------------
  # kernel matrix ONCE
  # ------------------------------------------------------------
  K <- kernel_mat(X, X, kernel, ls, m)
  
  # ------------------------------------------------------------
  # CV / OCV scores
  # ------------------------------------------------------------
  scores <- numeric(length(lambda_grid))
  
  for (i in seq_along(lambda_grid)) {
    
    lam <- lambda_grid[i]
    
    # ============================================================
    # LS: OCV (correct)
    # ============================================================
    if (loss == "ls") {
      
      scores[i] <- rkhs_ls_ocv(K, y, lam)
      
    }
    
    # ============================================================
    # QUANTILE: GCV (your smoothed version)
    # ============================================================
    else {
      
      fit_tmp <- rkhs_quantile_irls(K, y, tau = tau,
                                    lambda = lam)
      
      scores[i] <- rkhs_quantile_gcv(
        K,
        y,
        fit_tmp$alpha,
        fit_tmp$weights,
        lam
      )
    }
  }
  
  lambda <- lambda_grid[which.min(scores)]
  
  if (verbose) {
    cat("selected lambda:", format(lambda, scientific = TRUE), "\n")
  }
  
  # ============================================================
  # FINAL FIT
  # ============================================================
  if (loss == "ls") {
    
    fit <- rkhs_ls(K, y, lambda)
    
    return(list(
      alpha = fit$alpha,
      fitted = fit$fitted,
      lambda = lambda,
      kernel = kernel,
      loss = "ls"
    ))
  }
  
  fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lambda)
  
  list(
    alpha = fit$alpha,
    fitted = fit$fitted,
    weights = fit$weights,
    lambda = lambda,
    kernel = kernel,
    loss = "quantile",
    tau = tau,
    iterations = fit$iterations
  )
}

# ============================================================
# Prediction
# ============================================================
predict_rkhs <- function(fit, X_train, X_new,
                         kernel = fit$kernel,
                         m = 2.5,
                         ls = 1.0) {
  
  X_train <- as.matrix(X_train)
  X_new   <- as.matrix(X_new)
  
  Knew <- kernel_mat(X_train, X_new, kernel, ls, m)
  
  as.vector(t(Knew) %*% fit$alpha)
}
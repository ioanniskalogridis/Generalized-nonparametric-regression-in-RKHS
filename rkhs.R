setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_osqp.cpp")

# =============================================
# Main fitting function with GCV
# =============================================
fit_rkhs_quantile <- function(X, y, 
                              tau = 0.5, 
                              lambda = NULL,
                              kernel = "matern", 
                              length_scale = 1.0,
                              use_gcv = TRUE) {
  
  X_train <- as.matrix(X)
  n <- nrow(X_train)
  
  # Compute kernel matrix once
  K_train <- if(kernel == "gaussian") {
    gaussian_kernel(X_train, X_train, ls = length_scale)
  } else {
    matern52_kernel(X_train, X_train, ls = length_scale)
  }
  
  # ------------------- GCV for lambda -------------------
  if (is.null(lambda) && use_gcv) {
    lambda_grid <- 10^seq(-7, -1, length.out = 40)   # safer range
    
    gcv_scores <- numeric(length(lambda_grid))
    
    for(i in seq_along(lambda_grid)) {
      lam <- lambda_grid[i]
      
      fit_tmp <- rkhs_quantile_irls(K_train, y, tau = tau, lambda = lam)
      alpha_tmp <- as.numeric(fit_tmp$alpha)
      
      # Compute training loss (check loss)
      fitted <- sapply(1:n, function(i) sum(K_train[i, ] * alpha_tmp))
      residuals <- y - fitted
      train_loss <- mean(pmax(tau * residuals, (tau - 1) * residuals))
      
      # Approximate effective degrees of freedom
      A <- K_train + n * lam * diag(n)
      df <- sum(diag(solve(A, K_train)))
      
      penalty <- (1 - df / n)^2
      gcv_scores[i] <- train_loss/penalty #/ max(penalty, 1e-8)
    }
    
    best_idx <- which.min(gcv_scores)
    lambda <- lambda_grid[best_idx]
    cat("GCV selected lambda =", format(lambda, scientific = TRUE), "\n")
  }
  
  # ------------------- Final fit -------------------
  fit <- rkhs_quantile_irls(K_train, y, tau = tau, lambda = lambda)
  alpha <- as.numeric(fit$alpha)
  
  # Predictor function
  predictor <- function(X_new) {
    X_new <- as.matrix(X_new)
    n_new <- nrow(X_new)
    preds <- numeric(n_new)
    
    for(i in 1:n_new) {
      k_row <- if(kernel == "gaussian") {
        gaussian_kernel(X_train, X_new[i, , drop = FALSE], ls = length_scale)
      } else {
        matern52_kernel(X_train, X_new[i, , drop = FALSE], ls = length_scale)
      }
      preds[i] <- sum(k_row * alpha)
    }
    preds
  }
  
  list(alpha = alpha,
       predictor = predictor,
       fitted = sapply(1:n, function(i) sum(K_train[i, ] * alpha)),
       lambda = lambda,
       length_scale = length_scale,
       tau = tau,
       kernel = kernel)
}

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_osqp.cpp")

# =============================================
# Optimized RKHS Quantile Regression
# =============================================
fit_rkhs_quantile <- function(X, y,
                              tau = 0.5,
                              lambda = NULL,
                              kernel = c("matern", "gaussian"),
                              m = 2.5,
                              length_scale = 1.0,
                              use_gcv = TRUE) {
  
  kernel <- match.arg(kernel)
  
  X_train <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X_train)
  
  # Kernel matrix
  K_train <- if (kernel == "gaussian") {
    gaussian_kernel(X_train, X_train, ls = length_scale)
  } else {
    matern_kernel(X_train, X_train, m = m, ls = length_scale)
  }
  
  # GCV for lambda
  if (is.null(lambda) && use_gcv) {
    lambda_grid <- 10^seq(-6, -1, length.out = 25)
    gcv_scores <- numeric(length(lambda_grid))
    
    for (i in seq_along(lambda_grid)) {
      lam <- lambda_grid[i]
      fit_tmp <- rkhs_quantile_irls(K_train, y, tau = tau, lambda = lam)
      alpha_tmp <- as.numeric(fit_tmp$alpha)
      
      fitted <- as.vector(K_train %*% alpha_tmp)   # safe in-sample
      residuals <- y - fitted
      train_loss <- mean(pmax(tau * residuals, (tau - 1) * residuals))
      
      A <- K_train + n * lam * diag(n) + 1e-10 * diag(n)
      df <- sum(diag(solve(A, K_train)))
      
      penalty <- (1 - df / n)^2
      gcv_scores[i] <- train_loss / max(penalty, 1e-8)
    }
    
    best_idx <- which.min(gcv_scores)
    lambda <- lambda_grid[best_idx]
    cat("GCV selected lambda =", format(lambda, scientific = TRUE), "\n")
  }
  
  # Final fit
  fit <- rkhs_quantile_irls(K_train, y, tau = tau, lambda = lambda)
  alpha <- as.numeric(fit$alpha)
  
  # Fast predictor using C++
  predictor <- function(X_new) {
    X_new <- as.matrix(X_new)
    as.vector(rkhs_predict(X_train, X_new, alpha, 
                           kernel_type = kernel,
                           ls = length_scale,
                           m = m))
  }
  
  list(
    alpha        = alpha,
    predictor    = predictor,
    fitted       = as.vector(fit$fitted),
    lambda       = lambda,
    length_scale = length_scale,
    tau          = tau,
    kernel       = kernel,
    m            = if(kernel == "matern") m else NULL,
    iterations   = fit$iterations
  )
}

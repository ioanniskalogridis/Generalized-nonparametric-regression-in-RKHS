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
                              kernel = c("matern", "gaussian"),
                              m = NULL,
                              length_scale = 1.0,
                              use_gcv = TRUE) {
  
  kernel <- match.arg(kernel)
  
  X_train <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X_train)
  
  # =====================================================
  # Kernel matrix
  # =====================================================
  K_train <- if (kernel == "gaussian") {
    
    gaussian_kernel(X_train, X_train, ls = length_scale)
    
  } else {
    
    if (is.null(m))
      stop("For Matérn kernel, you must specify m (Sobolev order).")
    
    matern_kernel(X_train, X_train, m = m, ls = length_scale)
  }
  
  # =====================================================
  # GCV selection for lambda
  # =====================================================
  if (is.null(lambda) && use_gcv) {
    
    lambda_grid <- 10^seq(-7, -1, length.out = 30)
    
    gcv_scores <- numeric(length(lambda_grid))
    
    I_n <- diag(n)
    
    for (i in seq_along(lambda_grid)) {
      
      lam <- lambda_grid[i]
      
      fit_tmp <- rkhs_quantile_irls(K_train, y,
                                    tau = tau,
                                    lambda = lam)
      
      alpha_tmp <- as.numeric(fit_tmp$alpha)
      
      fitted <- K_train %*% alpha_tmp
      residuals <- y - fitted
      
      # pinball loss
      train_loss <- mean(pmax(tau * residuals,
                              (tau - 1) * residuals))
      
      # =====================================================
      # Approximate effective degrees of freedom:
      # S = K (K + nλ I)^(-1)
      # df = tr(S)
      # =====================================================
      A <- K_train + n * lam * I_n
      
      df <- sum(diag(solve(A, K_train)))
      
      gcv_scores[i] <- train_loss / (1 - df / n)^2
    }
    
    best_idx <- which.min(gcv_scores)
    lambda <- lambda_grid[best_idx]
    
    cat("GCV selected lambda =", format(lambda, scientific = TRUE), "\n")
  }
  
  # =====================================================
  # Final fit
  # =====================================================
  fit <- rkhs_quantile_irls(K_train, y,
                            tau = tau,
                            lambda = lambda)
  
  alpha <- as.numeric(fit$alpha)
  
  # =====================================================
  # Prediction function (vectorised per new block)
  # =====================================================
  predictor <- function(X_new) {
    
    X_new <- as.matrix(X_new)
    m_new <- nrow(X_new)
    
    preds <- numeric(m_new)
    
    for (i in seq_len(m_new)) {
      
      x_new <- X_new[i, , drop = FALSE]
      
      k_vec <- if (kernel == "gaussian") {
        gaussian_kernel(X_train, x_new, ls = length_scale)
      } else {
        matern_kernel(X_train, x_new, m = m, ls = length_scale)
      }
      
      preds[i] <- sum(k_vec * alpha)
    }
    
    preds
  }
  
  # =====================================================
  # In-sample fit
  # =====================================================
  fitted_vals <- as.vector(K_train %*% alpha)
  
  list(
    alpha = alpha,
    predictor = predictor,
    fitted = fitted_vals,
    lambda = lambda,
    length_scale = length_scale,
    tau = tau,
    kernel = kernel,
    m = m
  )
}
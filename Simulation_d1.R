library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(knitr)
library(kableExtra)

sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# 1D TARGET GENERATOR
# ============================================================

generate_f0_1d <- function(beta, K = 50, p = 1.2) {
  
  k <- 1:K
  coeff <- k^(-(2 * beta + p))
  
  f0 <- function(X) {
    
    X <- as.matrix(X)
    
    out <- rep(0, nrow(X))
    
    for (j in 1:K) {
      out <- out + coeff[j] * cos(2 * pi * j * X[,1])
    }
    
    out
  }
  
  list(f0 = f0, beta = beta)
}

# ============================================================
# SETTINGS
# ============================================================

B <- 500
n <- 200

beta_levels <- c(0.5, 0.75, 1.0)
error_types <- c("gaussian", "t2")

results <- list()

grid_fits <- list()
store_B <- 20

# ============================================================
# FIXED PREDICTION GRID
# ============================================================

Xg <- matrix(seq(0, 1, length = 300), ncol = 1)

# ============================================================
# SIMULATION
# ============================================================

for (beta in beta_levels) {
  
  for (err in error_types) {
    
    mse_ls <- numeric(B)
    mse_q  <- numeric(B)
    
    f_target <- generate_f0_1d(beta)
    
    f0g_fixed <- f_target$f0(Xg)
    
    for (b in 1:B) {
      
      if (b %% 10 == 0)
        cat("beta =", beta,
            " error =", err,
            " rep =", b, "\n")
      
      X <- matrix(runif(n), n, 1)
      
      f0 <- f_target$f0(X)
      
      eps <- if (err == "gaussian")
        rnorm(n)
      else
        rt(n, df = 2)
      
      y <- f0 + eps
      
      fit_ls <- fit_rkhs(
        X, y,
        loss = "ls",
        kernel = "matern",
        s = 2.5,
        ls = 1
      )
      
      fit_q <- fit_rkhs(
        X, y,
        loss = "quantile",
        tau = 0.5,
        kernel = "matern",
        s = 2.5,
        ls = 1
      )
      
      f_ls <- predict_rkhs(fit_ls, X, Xg)
      f_q  <- predict_rkhs(fit_q,  X, Xg)
      
      mse_ls[b] <- mean((f_ls - f0g_fixed)^2)
      mse_q[b]  <- mean((f_q  - f0g_fixed)^2)
      
      if (b <= store_B) {
        
        key <- paste0(
          "d1_beta", beta,
          "_", err,
          "_rep", b
        )
        
        grid_fits[[key]] <- list(
          d = 1,
          beta = beta,
          error = err,
          rep = b,
          Xg = Xg,
          f0 = f0g_fixed,
          f_ls = f_ls,
          f_q = f_q
        )
      }
    }
    
    results[[paste0(
      "d1_beta", beta,
      "_", err
    )]] <- data.frame(
      d = 1,
      beta = beta,
      error = err,
      mse_ls_mean = mean(mse_ls),
      mse_ls_sd = sd(mse_ls) / sqrt(B),
      mse_q_mean = mean(mse_q),
      mse_q_sd = sd(mse_q) / sqrt(B)
    )
    
    saveRDS(results, "results_d1_partial.rds")
    saveRDS(grid_fits, "gridfits_d1_partial.rds")
  }
}

saveRDS(results, "results_d1.rds")
saveRDS(grid_fits, "gridfits_d1.rds")
library(Rcpp)
library(RcppArmadillo)
sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# 1D and 2D Target Generators (Fixed f0 in [H]^β)
# ============================================================
generate_f0_1d <- function(beta, K = 60, p = 1.2) {
  k <- 1:K
  coeff <- k^(-(2 * beta + p))
  
  f0 <- function(X) {
    X <- as.matrix(X)
    out <- rep(0, nrow(X))
    for (j in 1:K) {
      out <- out + coeff[j] * cos(2 * pi * j * X[,1])
    }
  }
  list(f0 = f0, beta = beta)
}

generate_f0_2d <- function(beta, K = 25, p = 1.2) {
  f0 <- function(X) {
    X <- as.matrix(X)
    out <- rep(0, nrow(X))
    for (j in 1:K) {
      for (k in 1:K) {
        coeff <- (j * k)^(-(2 * beta + p))
        out <- out + coeff * cos(2 * pi * j * X[,1]) * cos(2 * pi * k * X[,2])
      }
    }
  }
  list(f0 = f0, beta = beta)
}

# ============================================================
# SIMULATION
# ============================================================
B <- 60
n <- 250
beta_levels <- c(0.5, 0.75, 1.0)
error_types <- c("gaussian", "t2")

results <- list()

for (d in c(1, 2)) {
  for (beta in beta_levels) {
    for (err in error_types) {
      
      mse_ls <- numeric(B)
      mse_q  <- numeric(B)
      
      f_target <- if (d == 1) generate_f0_1d(beta) else generate_f0_2d(beta)
      
      for (b in 1:B) {
        if (b %% 10 == 0) cat("d=", d, " beta=", beta, " err=", err, " rep=", b, "\n")
        
        # Data
        X <- matrix(runif(n * d), n, d)
        f0 <- f_target$f0(X)
        eps <- if (err == "gaussian") rnorm(n, sd = 0.5) else rt(n, df = 2) * 0.5
        y <- f0 + eps
        
        # Fits
        fit_ls <- fit_rkhs(X, y, loss = "ls", 
                           kernel = if(d==1) "matern" else "tensor",
                           s = 2.5, ls = 0.8)
        
        fit_q <- fit_rkhs(X, y, loss = "quantile", tau = 0.5, 
                          kernel = if(d==1) "matern" else "tensor",
                          s = 2.5, ls = 0.8)
        
        # Grid
        if (d == 1) {
          Xg <- matrix(seq(0, 1, length = 300), ncol = 1)
        } else {
          x1 <- seq(0, 1, length = 50)
          x2 <- seq(0, 1, length = 50)
          Xg <- as.matrix(expand.grid(x1, x2))
        }
        
        f0g <- f_target$f0(Xg)
        
        # Predict
        f_ls <- predict_rkhs(fit_ls, X, Xg)
        f_q  <- predict_rkhs(fit_q,  X, Xg)
        
        mse_ls[b] <- mean((f_ls - f0g)^2)
        mse_q[b]  <- mean((f_q - f0g)^2)
      }
      
      results[[paste0("d", d, "_beta", beta, "_", err)]] <- 
        data.frame(
          d = d,
          beta = beta,
          error = err,
          mse_ls_mean = mean(mse_ls),
          mse_ls_sd = sd(mse_ls),
          mse_q_mean = mean(mse_q),
          mse_q_sd = sd(mse_q)
        )
    }
  }
}

summary_df <- do.call(rbind, results)
print(summary_df)
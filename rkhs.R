setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")

library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_quan.cpp")

# ============================================================
# RKHS FIT WRAPPER (clean + minimal)
# ============================================================
fit_rkhs <- function(X, y,
                      loss = c("ls", "quantile"),
                      kernel = c("gaussian", "matern", "tensor"),
                      tau = 0.5,
                      s = 2.5,
                      ls = 1.0,
                      lambda_grid = 10^seq(-08, -1, length.out = 50),
                      cv = TRUE,
                      verbose = FALSE) {

  loss   <- match.arg(loss)
  kernel <- match.arg(kernel)

  X <- as.matrix(X)
  y <- as.numeric(y)

  n <- nrow(X)

  # ============================================================
  # CV SCORE STORAGE
  # ============================================================
  scores <- numeric(length(lambda_grid))

  # ============================================================
  # MODEL SELECTION LOOP
  # ============================================================
  for (i in seq_along(lambda_grid)) {

    lam <- lambda_grid[i]

    # ---------------- LS ----------------
    if (loss == "ls") {

      K <- kernel_mat(X, X, kernel, ls, s)
      fit <- rkhs_ls(K, y, lam)

      scores[i] <- rkhs_ls_ocv(K, y, lam)
    }

    # ---------------- QUANTILE ----------------
    if (loss == "quantile") {

      K <- kernel_mat(X, X, kernel, ls, s)
      fit <- rkhs_quantile_irls(K, y, tau = tau, lambda = lam)
      alpha <- fit$alpha
      w <- fit$weights

      scores[i] <- rkhs_quantile_gcv(K,y, alpha, w, lam)
    }
  }
  # print(scores)

  lambda <- lambda_grid[which.min(scores)]

  if (verbose) {
    cat("Selected lambda:", format(lambda, scientific = TRUE), "\n")
  }

  # ============================================================
  # FINAL FIT
  # ============================================================
  K <- kernel_mat(X, X, kernel, ls, s)

  if (loss == "ls") {

    fit <- rkhs_ls(K, y, lambda)

    return(list(
      alpha = fit$alpha,
      fitted = fit$fitted,
      lambda = lambda,
      kernel = kernel,
      loss = "ls",
      s = s
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
    s = s
  )
}

predict_rkhs <- function(fit, X_train, X_new,
                         kernel = fit$kernel,
                         s = fit$s,
                         ls = 1.0) {
  
  Knew <- kernel_mat(X_train, X_new, kernel, ls, s)
  
  as.vector(t(Knew) %*% fit$alpha)
}

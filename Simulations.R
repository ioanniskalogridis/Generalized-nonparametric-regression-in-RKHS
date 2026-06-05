library(Rcpp)
library(RcppArmadillo)
sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# SETTINGS
# ============================================================
B <- 30                    # Start with small B for testing
n <- 200
grid_size <- 150
x_grid <- seq(-1, 1, length.out = grid_size)
X_grid <- matrix(x_grid, ncol = 1)

mse_ls <- numeric(B)
mse_q  <- numeric(B)
pred_ls_all <- vector("list", B)
pred_q_all  <- vector("list", B)

# ============================================================
# SIMULATION
# ============================================================
for (b in 1:B) {
  if (b %% 5 == 0) cat("Simulation", b, "of", B, "\n")
  
  # Training data
  X <- matrix(runif(n, -1, 1), n, 1)
  f0 <- sin(2*pi*X[,1]) + 0.3*X[,1]^2
  y <- f0 + 1 * rnorm(n)
  
  # LS Fit
  fit_ls <- fit_rkhs(X, y, loss = "ls", kernel = "matern", s = 2.5, ls = 0.8)
  
  # Quantile Fit
  fit_q <- fit_rkhs(X, y, loss = "quantile", tau = 0.5, 
                    kernel = "matern", s = 2.5, ls = 0.8)
  
  # Predictions - EXACTLY as your function expects
  f_ls_grid <- predict_rkhs(fit_ls, X, X_grid)      # fit, X_train, X_new
  f_q_grid  <- predict_rkhs(fit_q,  X, X_grid)
  
  # True function
  f0_grid <- sin(2*pi*x_grid) + 0.3*x_grid^2
  
  # Store
  mse_ls[b] <- mean((f_ls_grid - f0_grid)^2)
  mse_q[b]  <- mean((f_q_grid  - f0_grid)^2)
  
  pred_ls_all[[b]] <- f_ls_grid
  pred_q_all[[b]]  <- f_q_grid
}

# ============================================================
# RESULTS
# ============================================================
cat("\n=== SIMULATION RESULTS (B =", B, ") ===\n")
cat("LS      - Mean MSE:", round(mean(mse_ls), 5), 
    " | Median:", round(median(mse_ls), 5), "\n")
cat("Quantile - Mean MSE:", round(mean(mse_q), 5), 
    " | Median:", round(median(mse_q), 5), "\n")

# ============================================================
# VISUALIZATION
# ============================================================
B_show <- min(15, B)

ylim_range <- range(c(f0_grid, 
                      unlist(pred_ls_all[1:B_show]), 
                      unlist(pred_q_all[1:B_show])))

plot(x_grid, f0_grid, type = "l", lwd = 3, col = "black",
     ylim = ylim_range, xlab = "x", ylab = "f(x)",
     main = paste0("RKHS Simulation (B=", B, ", n=", n, ")"))

for (b in 1:B_show) {
  lines(x_grid, pred_ls_all[[b]], col = rgb(1, 0, 0, 0.25), lwd = 1)
  lines(x_grid, pred_q_all[[b]],  col = rgb(0, 0, 1, 0.25), lwd = 1)
}

legend("topright", 
       legend = c("True f0", "LS (red)", "Quantile (blue)"),
       col = c("black", "red", "blue"), 
       lwd = c(3, 1, 1), bty = "n")
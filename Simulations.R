library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# SETTINGS
# ============================================================
B <- 200
n <- 100
B_show <- B

grid_size <- 100
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
  print(b)
  # ---------------- training data ----------------
  X <- matrix(runif(n, -1, 1), n, 1)
  
  f0 <- sin(2*pi*X[,1]) + 0.3*X[,1]^2
  y  <- f0 + 0.5 * rnorm(n)
  
  # ============================================================
  # FIT LS (internal CV)
  # ============================================================
  fit_ls <- fit_rkhs(X, y, loss = "ls", kernel = "matern")
  
  f_ls_grid <- predict_rkhs(fit_ls, X, X_grid)
  
  # ============================================================
  # FIT QUANTILE (internal CV)
  # ============================================================
  fit_q <- fit_rkhs(X, y, loss = "quantile", kernel = "matern", tau = 0.5)
  
  f_q_grid <- predict_rkhs(fit_q, X, X_grid)
  
  # ============================================================
  # TRUE FUNCTION ON GRID
  # ============================================================
  f0_grid <- sin(2*pi*x_grid) + 0.3*x_grid^2
  
  # ============================================================
  # MSE ON GRID (THIS IS NOW FUNCTIONAL ERROR)
  # ============================================================
  mse_ls[b] <- mean((f_ls_grid - f0_grid)^2)
  mse_q[b]  <- mean((f_q_grid  - f0_grid)^2)
  
  # store
  pred_ls_all[[b]] <- f_ls_grid
  pred_q_all[[b]]  <- f_q_grid
}

# ============================================================
# RESULTS
# ============================================================
cat("\nLS mean:", mean(mse_ls),
    "median:", median(mse_ls), "\n")

cat("Q mean:", mean(mse_q),
    "median:", median(mse_q), "\n")

# true function on grid
f0_grid <- sin(2*pi*x_grid) + 0.3*x_grid^2

# ---------------- set plotting range ----------------
ylim_range <- range(c(
  f0_grid,
  unlist(pred_ls_all[1:B_show]),
  unlist(pred_q_all[1:B_show])
))

# ---------------- base plot ----------------
plot(x_grid, f0_grid,
     type = "l",
     lwd = 3,
     col = "black",
     ylim = ylim_range,
     xlab = "x",
     ylab = "f(x)",
     main = "RKHS Simulation (LS vs Quantile, Grid Evaluation)")

# ---------------- overlay sample functions ----------------
for (b in 1:B_show) {
  
  # LS (blue)
  lines(x_grid,
        pred_ls_all[[b]],
        col = rgb(0, 0, 1, 0.25),
        lwd = 1)
  
  # Quantile (red)
  lines(x_grid,
        pred_q_all[[b]],
        col = rgb(1, 0, 0, 0.25),
        lwd = 1)
}

# ---------------- legend ----------------
legend("topright",
       legend = c("True f0", "LS", "Quantile"),
       col = c("black", "blue", "red"),
       lwd = c(3, 1, 1))

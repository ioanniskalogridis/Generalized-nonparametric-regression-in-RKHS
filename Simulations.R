library(Rcpp)
library(RcppArmadillo)

sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# SETTINGS
# ============================================================
B <- 200
n <- 300
B_show <- 20

mse_ls <- numeric(B)
mse_q  <- numeric(B)

pred_ls_all <- vector("list", B)
pred_q_all  <- vector("list", B)

# ============================================================
# SIMULATION
# ============================================================
for (b in 1:B) {
  
  # ---------------- data ----------------
  X <- matrix(runif(n, -1, 1), n, 1)
  
  f0 <- sin(2*pi*X[,1]) + 0.3*X[,1]^2
  
  y <- f0 + 0.5 * rnorm(n)
  
  # ============================================================
  # LS: EVERYTHING INTERNAL (X, y ONLY)
  # ============================================================
  fit_ls <- fit_rkhs(
    X = X,
    y = y,
    loss = "ls",
    kernel = "matern",
    cv = TRUE
  )
  
  f_ls <- as.numeric(fit_ls$fitted)
  
  # ============================================================
  # QUANTILE: EVERYTHING INTERNAL (X, y ONLY)
  # ============================================================
  fit_q <- fit_rkhs(
    X = X,
    y = y,
    loss = "quantile",
    tau = 0.5,
    kernel = "matern",
    cv = TRUE
  )
  
  f_q <- as.numeric(fit_q$fitted)
  
  # ============================================================
  # TRUE MSE (NO CV CONTAMINATION)
  # ============================================================
  mse_ls[b] <- mean((f_ls - f0)^2)
  mse_q[b]  <- mean((f_q  - f0)^2)
  
  # store for plotting
  pred_ls_all[[b]] <- f_ls
  pred_q_all[[b]]  <- f_q
}

# ============================================================
# RESULTS
# ============================================================
cat("\nLS mean:", mean(mse_ls),
    "median:", median(mse_ls), "\n")

cat("Q mean:", mean(mse_q),
    "median:", median(mse_q), "\n")

x_plot <- matrix(runif(n, -1, 1), n, 1)
ord <- order(x_plot[,1])
x_sorted <- x_plot[ord,1]

f0_plot <- sin(2*pi*x_sorted) + 0.3*x_sorted^2

ylim_range <- range(c(
  f0_plot,
  unlist(pred_ls_all[1:B_show]),
  unlist(pred_q_all[1:B_show])
))

plot(x_sorted, f0_plot,
     type = "l",
     lwd = 3,
     col = "black",
     ylim = ylim_range,
     xlab = "x",
     ylab = "f(x)",
     main = "RKHS Simulation (Fully Internal CV)")

for (b in 1:B_show) {
  
  lines(x_sorted,
        pred_ls_all[[b]][ord],
        col = rgb(0, 0, 1, 0.25),
        lwd = 1)
  
  lines(x_sorted,
        pred_q_all[[b]][ord],
        col = rgb(1, 0, 0, 0.25),
        lwd = 1)
}

legend("topright",
       legend = c("True f0", "LS", "Quantile"),
       col = c("black", "blue", "red"),
       lwd = c(3,1,1))
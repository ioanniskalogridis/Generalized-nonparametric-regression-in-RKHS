# =============================================
# SMALL TEST EXAMPLE
# =============================================

# 2. Generate small 1D dataset
n <- 300
X <- matrix(runif(n), nrow = n, ncol = 1)

f_true <- function(x) sin(2 * pi * x) + 0.5 * x^2
y <- f_true(X) + rnorm(n, sd = 0.6)

cat("Data generated: n =", n, "\n")

# 3. Fit the model
fit <- fit_rkhs_quantile(
  X = X, 
  y = y,
  tau = 0.5,
  kernel = "gaussian",
  m = 2.5,
  length_scale = 0.8,
  use_gcv = TRUE
)

# 4. Results
cat("\n--- Results ---\n")
cat("Kernel:           ", fit$kernel, "\n")
cat("Length scale:     ", fit$length_scale, "\n")
cat("Lambda (GCV):     ", format(fit$lambda, scientific = TRUE), "\n")
cat("Iterations:       ", fit$iterations, "\n")
cat("Training RMSE:    ", round(sqrt(mean((y - fit$fitted)^2)), 4), "\n")

# 5. Prediction on new points
X_new <- matrix(seq(0, 1, length.out = 200), ncol = 1)
preds <- fit$predictor(X_new)

cat("Prediction successful! Length =", length(preds), "\n")

# 6. Plot
plot(X[,1], y, pch = 16, col = rgb(0,0,0,0.4), 
     main = "RKHS Quantile Regression (τ = 0.5)",
     xlab = "x", ylab = "y", cex = 0.8)

lines(X_new[,1], f_true(X_new), col = "black", lwd = 2, lty = 2)   # True function
lines(X_new[,1], preds, col = "blue", lwd = 2.5)                   # Estimated

legend("topright", 
       legend = c("Observations", "True function", "Estimated (τ=0.5)"),
       col = c(rgb(0,0,0,0.4), "black", "blue"), 
       pch = c(16, NA, NA), lty = c(NA, 2, 1), lwd = c(1,2,2.5))

nsim <- 200

n <- 200
X <- matrix(runif(n), nrow = n, ncol = 1)

f_true <- function(x) sin(2 * pi * x) + 0.5 * x^2
y <- f_true(X) + rt(n, df = 1) #+rnorm(n, sd = 0.6)



fit <- fit_rkhs_quantile(
  X = X, 
  y = y,
  tau = 0.5,
  kernel = "gaussian",
  m = 2.5,
  length_scale = 1,
  use_gcv = TRUE
)

X_new <- matrix(seq(0, 1, length.out = 200), ncol = 1)
preds <- fit$predictor(X_new)

cat("Prediction successful! Length =", length(preds), "\n")

# 6. Plot
plot(X[,1], y, pch = 16, col = rgb(0,0,0,0.4), ylim = c(-2, 2),
     main = "RKHS Quantile Regression (τ = 0.5)",
     xlab = "x", ylab = "y", cex = 0.8)

lines(X_new[,1], f_true(X_new), col = "black", lwd = 2, lty = 2)   # True function
lines(X_new[,1], preds, col = "blue", lwd = 2.5)                   # Estimated

legend("topright", 
       legend = c("Observations", "True function", "Estimated (τ=0.5)"),
       col = c(rgb(0,0,0,0.4), "black", "blue"), 
       pch = c(16, NA, NA), lty = c(NA, 2, 1), lwd = c(1,2,2.5))

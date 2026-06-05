library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(knitr)
library(kableExtra)

sourceCpp("rkhs_quan.cpp")

set.seed(123)

# ============================================================
# 1D and 2D Target Generators
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
    return(out)
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
        out <- out + coeff * sin(2 * pi * j * X[,1]) *sin(2 * pi * k * X[,2])
      }
    }
    return(out)
  }
  list(f0 = f0, beta = beta)
}

# ============================================================
# SIMULATION SETTINGS
# ============================================================

B <- 500
n <- 200
beta_levels <- c(0.5, 0.75, 1.0)
error_types <- c("gaussian", "t2")

results <- list()

# ============================================================
# NEW: STORE GRID FITS
# ============================================================

grid_fits <- list()
store_B <- 20   # first 20 reps

# ============================================================
# GRID (computed once per dimension)
# ============================================================

Xg_list <- list(
  "1" = matrix(seq(0, 1, length = 300), ncol = 1),
  "2" = as.matrix(expand.grid(
    seq(0, 1, length = 50),
    seq(0, 1, length = 50)
  ))
)

# ============================================================
# SIMULATION
# ============================================================

for (d in c(1, 2)) {
  for (beta in beta_levels) {
    for (err in error_types) {
      
      mse_ls <- numeric(B)
      mse_q  <- numeric(B)
      
      f_target <- if (d == 1) generate_f0_1d(beta) else generate_f0_2d(beta)
      
      Xg <- Xg_list[[as.character(d)]]
      f0g_fixed <- f_target$f0(Xg)
      
      for (b in 1:B) {
        
        if (b %% 10 == 0)
          cat("d=", d, " beta=", beta, " err=", err, " rep=", b, "\n")
        
        # ----------------------------
        # DATA
        # ----------------------------
        X <- matrix(runif(n * d), n, d)
        f0 <- f_target$f0(X)
        
        eps <- if (err == "gaussian") rnorm(n, sd = 0.5) else rt(n, df = 2) * 0.5
        y <- f0 + eps
        
        # ----------------------------
        # FITS
        # ----------------------------
        fit_ls <- fit_rkhs(
          X, y, loss = "ls",
          kernel = if(d==1) "matern" else "tensor",
          s = 2.5, ls = 1
        )
        
        fit_q <- fit_rkhs(
          X, y, loss = "quantile", tau = 0.5,
          kernel = if(d==1) "matern" else "tensor",
          s = 2.5, ls = 1
        )
        
        # ----------------------------
        # PREDICTION
        # ----------------------------
        f_ls <- predict_rkhs(fit_ls, X, Xg)
        f_q  <- predict_rkhs(fit_q,  X, Xg)
        
        mse_ls[b] <- mean((f_ls - f0g_fixed)^2)
        mse_q[b]  <- mean((f_q  - f0g_fixed)^2)
        

        # Store 20 curves fot plotting
        if (b <= store_B) {
          
          key <- paste0("d", d, "_beta", beta, "_", err, "_rep", b)
          
          grid_fits[[key]] <- list(
            d = d,
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
      
      # ----------------------------
      # SAVE SUMMARY RESULTS
      # ----------------------------
      results[[paste0("d", d, "_beta", beta, "_", err)]] <-
        data.frame(
          d = d,
          beta = beta,
          error = err,
          mse_ls_mean = mean(mse_ls),
          mse_ls_sd = sd(mse_ls) / sqrt(B),
          mse_q_mean = mean(mse_q),
          mse_q_sd = sd(mse_q) / sqrt(B)
        )
    }
  }
}

df <- do.call(rbind, lapply(results, as.data.frame))

rownames(df) <- NULL

df <- df %>%
  mutate(
    d = as.numeric(d),
    beta = sprintf("%.2f", beta),
    error = ifelse(error == "gaussian", "Gaussian", "t2")
  ) %>%
  arrange(d, error, beta)

df <- df %>%
  select(
    d,
    error,
    beta,
    mse_ls_mean, mse_ls_sd,
    mse_q_mean, mse_q_sd
  )

df <- df %>%
  mutate(mse_ls_mean = 10*mse_ls_mean, mse_ls_sd = 10* mse_ls_sd,
         mse_q_mean = 10*mse_q_mean, mse_q_sd = 10*mse_q_sd)

# Export table to latex

latex_out <- kable(
  df,
  format = "latex",
  booktabs = TRUE,
  escape = TRUE,
  digits = 4,
  align = c("c","c","c","r","r","r","r"),
  col.names = c(
    "$d$",
    "Noise",
    "$\\beta$",
    "MSE", "SE",
    "MSE", "SE"
  )
) %>%
  add_header_above(c(
    " " = 3,
    "Least Squares" = 2,
    "Quantile ($\\tau=0.5$)" = 2
  )) %>%
  kable_styling(
    latex_options = c("hold_position", "scale_down"),
    full_width = FALSE
  )

cat(latex_out)

# Plotting (Figures 1 and 2 in the paper)

keys <- names(grid_fits)
sel <- keys[grepl("d1_beta0.5_gaussian", keys)]
objs <- grid_fits[sel]

obj0 <- objs[[1]]

plot(obj0$Xg, obj0$f0, type = "l",
     col = "black", lwd = 2,
     ylim = range(c(obj0$f0, obj0$f_ls, obj0$f_q)),
     xlab = "x", ylab = "f(x)",
     main = "d = 1: All sample curves (LS = blue, QR = red)")


for (i in seq_along(objs)) {
  obj <- objs[[i]]
  
  lines(obj$Xg, obj$f_ls, col = rgb(0, 0, 1, 0.25))  # LS transparent blue
  lines(obj$Xg, obj$f_q,  col = rgb(1, 0, 0, 0.25))  # QR transparent red
}

legend("topright",
       legend = c("Truth", "LS", "QR"),
       col = c("black", "blue", "red"),
       lwd = 2)

# d = 2 
obj <- grid_fits[["d2_beta0.5_gaussian_rep1"]]
n = 50 # grid size
z_ls <- matrix(obj$f_ls, n, n, byrow = FALSE)
z_q  <- matrix(obj$f_q,  n, n, byrow = FALSE)
z_0  <- matrix(obj$f0,   n, n, byrow = FALSE)

image(z_ls, col = heat.colors(50),
      main = "")
contour(z_ls, add = TRUE)

image(z_q, col = heat.colors(50))
contour(z_q, add = TRUE)

image(z_0, col = heat.colors(50),
      main = "")
contour(z_0, add = TRUE)

library(ggplot2)

n <- 50
grid <- expand.grid(
  x1 = seq(0, 1, length = n),
  x2 = seq(0, 1, length = n)
)

df <- data.frame(
  x1 = grid$x1,
  x2 = grid$x2,
  z0 = obj$f0,
  zls = obj$f_ls,
  zq  = obj$f_q
)
ggplot(df, aes(x1, x2, fill = zls)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "Least Squares RKHS", x = "x1", y = "x2", fill = "f(x)") +
  theme_minimal()
ggplot(df, aes(x1, x2, fill = zq)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "Quantile RKHS", x = "x1", y = "x2", fill = "f(x)") +
  theme_minimal()
ggplot(df, aes(x1, x2, fill = z0)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "True function", x = "x1", y = "x2", fill = "f(x)") +
  theme_minimal()
ggplot(df, aes(x1, x2, fill = zls)) +
  geom_tile(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "LS RKHS", x = "x1", y = "x2", fill = "f(x)") +
  theme_minimal()

ggplot(df, aes(x1, x2, z = zls)) +
  geom_contour_filled(bins = 12) +
  scale_fill_viridis_d(option = "C", guide = "none") +
  geom_contour(color = "black", alpha = 0.3, bins = 12) +
  coord_fixed() +
  labs(title = "LS RKHS", x = "x1", y = "x2") +
  theme_minimal()
ggplot(df, aes(x1, x2, z = zq)) +
  geom_contour_filled(bins = 12) +
  scale_fill_viridis_d(option = "C", guide = "none") +
  geom_contour(color = "black", alpha = 0.3, bins = 12) +
  coord_fixed() +
  labs(title = "LS RKHS", x = "x1", y = "x2") +
  theme_minimal()
ggplot(df, aes(x = x1, y = x2, fill = zls)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "", x = "x1", y = "x2") +
  theme_minimal()
ggplot(df, aes(x = x1, y = x2, fill = zq)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "", x = "x1", y = "x2") +
  theme_minimal()
ggplot(df, aes(x = x1, y = x2, fill = z0)) +
  geom_raster(interpolate = TRUE) +
  scale_fill_viridis_c(option = "C") +
  coord_fixed() +
  labs(title = "", x = "x1", y = "x2") +
  theme_minimal()

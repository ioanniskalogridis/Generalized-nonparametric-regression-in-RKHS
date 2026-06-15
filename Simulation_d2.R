library(Rcpp)
library(RcppArmadillo)
library(dplyr)
library(knitr)
library(kableExtra)

sourceCpp("rkhs_quan.cpp")

set.seed(456)

# ============================================================
# 2D TARGET GENERATOR
# ============================================================

generate_f0_2d <- function(beta, K = 25, p = 2/3) {
  
  f0 <- function(X) {
    
    X <- as.matrix(X)
    
    out <- rep(0, nrow(X))
    
    for (j in 1:K) {
      for (k in 1:K) {
        
        coeff <- (j * k)^(-(2 * beta + p))
        
        out <- out +
          coeff *
          sin(2 * pi * j * X[,1]) *
          sin(2 * pi * k * X[,2])
      }
    }
    
    out
  }
  
  list(f0 = f0, beta = beta)
}

# ============================================================
# SETTINGS
# ============================================================

B <- 300
n <- 200

beta_levels <- c(0.5, 0.75, 1.0)
error_types <- c("gaussian", "t2")

results <- list()

grid_fits <- list()
store_B <- 20

# ============================================================
# FIXED PREDICTION GRID
# ============================================================

Xg <- as.matrix(
  expand.grid(
    seq(0, 1, length = 50),
    seq(0, 1, length = 50)
  )
)

# ============================================================
# SIMULATION
# ============================================================

for (beta in beta_levels) {
  
  for (err in error_types) {
    
    mse_ls <- numeric(B)
    mse_q  <- numeric(B)
    
    f_target <- generate_f0_2d(beta)
    
    f0g_fixed <- f_target$f0(Xg)
    
    for (b in 1:B) {
      
      if (b %% 10 == 0)
        cat("beta =", beta,
            " error =", err,
            " rep =", b, "\n")
      
      X <- matrix(runif(n * 2), n, 2)
      
      f0 <- f_target$f0(X)
      
      eps <- if (err == "gaussian")
        rnorm(n)
      else
        rt(n, df = 2)
      
      y <- f0 + eps
      
      fit_ls <- fit_rkhs(
        X, y,
        loss = "ls",
        kernel = "tensor",
        s = 2.5,
        ls = 1
      )
      
      fit_q <- fit_rkhs(
        X, y,
        loss = "quantile",
        tau = 0.5,
        kernel = "tensor",
        s = 2.5,
        ls = 1
      )
      
      f_ls <- predict_rkhs(fit_ls, X, Xg)
      f_q  <- predict_rkhs(fit_q,  X, Xg)
      
      mse_ls[b] <- mean((f_ls - f0g_fixed)^2)
      mse_q[b]  <- mean((f_q  - f0g_fixed)^2)
      
      if (b <= store_B) {
        
        key <- paste0(
          "d2_beta", beta,
          "_", err,
          "_rep", b
        )
        
        grid_fits[[key]] <- list(
          d = 2,
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
      "d2_beta", beta,
      "_", err
    )]] <- data.frame(
      d = 2,
      beta = beta,
      error = err,
      mse_ls_mean = mean(mse_ls),
      mse_ls_sd = sd(mse_ls) / sqrt(B),
      mse_q_mean = mean(mse_q),
      mse_q_sd = sd(mse_q) / sqrt(B)
    )
    
    saveRDS(results, "results_d2_partial.rds")
    saveRDS(grid_fits, "gridfits_d2_partial.rds")
  }
}

saveRDS(results, "results_d2.rds")
saveRDS(grid_fits, "gridfits_d2.rds")

# Combine results
results_d1 <- readRDS("results_d1.rds")
results_d2 <- readRDS("results_d2.rds")

results <- unname(c(results_d1, results_d2))
df <- as.data.frame(do.call(rbind, results))

# Make table
df <- df %>%
  mutate(
    d = as.integer(d),
    beta = as.numeric(beta),
    
    noise = ifelse(error %in% c("gaussian", "Gaussian"), "Gaussian", "$t_2$")
  ) %>%
  mutate(
    noise = factor(noise, levels = c("Gaussian", "$t_2$"))
  ) %>%
  arrange(d, beta, noise)


df <- df %>%
  mutate(
    mse_ls_mean = 10 * mse_ls_mean,
    mse_ls_sd   = 10 * mse_ls_sd,
    mse_q_mean  = 10 * mse_q_mean,
    mse_q_sd    = 10 * mse_q_sd
  )


df <- df %>%
  group_by(d, beta) %>%
  mutate(beta_n = n()) %>%
  ungroup() %>%
  group_by(d) %>%
  mutate(d_n = n()) %>%
  ungroup()

df <- df %>%
  mutate(
    d_col = ifelse(!duplicated(d),
                   paste0("\\multirow{", d_n, "}{*}{$d = ", d, "$}"),
                   ""),
    
    beta_col = ifelse(!duplicated(interaction(d, beta)),
                      paste0("\\multirow{", beta_n, "}{*}{$\\beta = ", sprintf("%.2f", beta), "$}"),
                      "")
  )


tab <- df %>%
  select(
    d_col,
    beta_col,
    noise,
    mse_ls_mean, mse_ls_sd,
    mse_q_mean, mse_q_sd
  )


latex_out <- kable(
  tab,
  format = "latex",
  booktabs = TRUE,
  escape = FALSE,
  digits = 4,
  align = c("c","c","c","r","r","r","r"),
  col.names = c(
    "$d$",
    "$\\beta$",
    "Noise",
    "LS MSE", "LS SE",
    "QR MSE", "QR SE"
  )
) %>%
  add_header_above(c(
    " " = 3,
    "Least Squares RKHS" = 2,
    "Quantile RKHS ($\\tau = 0.5$)" = 2
  )) %>%
  kable_styling(
    latex_options = c("hold_position", "scale_down"),
    full_width = FALSE,
    font_size = 10
  )

cat(latex_out)

# Plotting
# First one-dimensional estimates

library(dplyr)
library(ggplot2)

grid_d1 <- readRDS("gridfits_d1.rds")
extract_one_setting <- function(grid_list, target_beta, target_noise) {
  
  df_list <- lapply(grid_list, function(obj) {
    
    if (obj$beta != target_beta) return(NULL)
    if (obj$error != target_noise) return(NULL)
    
    data.frame(
      x  = obj$Xg[,1],
      ls = obj$f_ls,
      qr = obj$f_q,
      f0 = obj$f0,
      beta = obj$beta,
      noise = obj$error,
      rep = obj$rep
    )
  })
  
  bind_rows(df_list)
}

df_gauss <- extract_one_setting(grid_d1, 0.5, "gaussian")

margin.figs <- 0.01
setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
pdf("Fig1ls.pdf", width = 5, height = 3.9)
p_gauss_ls <- ggplot(df_gauss, aes(x = x, y = ls, group = rep)) +
  geom_line(colour = "grey70", linewidth = 0.65) +
  geom_line(aes(y = f0), colour = "black", linewidth = 1) +
  labs(title = "", y = "", x = "") +
  coord_cartesian(ylim = c(-1.2, 2)) + 
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(margin.figs, margin.figs, margin.figs, margin.figs, unit = "mm"),
    # axis.ticks = element_line(linewidth = 2, colour = "black"),
    axis.text = element_text(size = 13, colour = "black"),
    axis.ticks.length = unit(1.5, "mm")
  )
p_gauss_ls
dev.off()

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
pdf("Fig1lad.pdf", width = 5, height = 3.9)
p_gauss_qr <- ggplot(df_gauss, aes(x = x, y = qr, group = rep)) +
  geom_line(colour = "grey70", linewidth = 0.65) +
  geom_line(aes(y = f0), colour = "black", linewidth = 1) +
  labs(title = "", y = "", x = "") +
  coord_cartesian(ylim = c(-1.2, 2)) + 
  theme_minimal(base_size = 14) + 
  theme(
    plot.margin = margin(margin.figs, margin.figs, margin.figs, margin.figs, unit = "mm"),
    # axis.ticks = element_line(linewidth = 2, colour = "black"),
    axis.text = element_text(size = 13, colour = "black"),
    axis.ticks.length = unit(1.5, "mm")
  )
p_gauss_qr
dev.off()

df_t2 <- extract_one_setting(grid_d1, 0.5, "t2")

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
pdf("Fig2ls.pdf", width = 5, height = 3.9)
p_t2_ls <- ggplot(df_t2, aes(x = x, y = ls, group = rep)) +
  geom_line(colour = "grey70", linewidth = 0.65) +
  geom_line(aes(y = f0), colour = "black", linewidth = 1) +
  labs(title = "", y = "", x = "") +
  coord_cartesian(ylim = c(-1.2, 2)) + 
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(margin.figs, margin.figs, margin.figs, margin.figs, unit = "mm"),
    # axis.ticks = element_line(linewidth = 2, colour = "black"),
    axis.text = element_text(size = 13, colour = "black"),
    axis.ticks.length = unit(1.5, "mm")
  )
p_t2_ls
dev.off()

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
pdf("Fig2lad.pdf", width = 5, height = 3.9)
p_t2_qr <- ggplot(df_t2, aes(x = x, y = qr, group = rep)) +
  geom_line(colour = "grey70", linewidth = 0.65) +
  geom_line(aes(y = f0), colour = "black", linewidth = 1) +
  labs(title = "", y = "", x = "") +
  coord_cartesian(ylim = c(-1.2, 2)) + 
  theme_minimal(base_size = 14) +
  theme(
    plot.margin = margin(margin.figs, margin.figs, margin.figs, margin.figs, unit = "mm"),
    # axis.ticks = element_line(linewidth = 2, colour = "black"),
    axis.text = element_text(size = 13, colour = "black"),
    axis.ticks.length = unit(1.5, "mm")
  )
p_t2_qr
dev.off()

# Two-dimensional estimates now - just representatives

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Documents/GitHub/Generalized-nonparametric-regression-in-RKHS")
grid_d2 <- readRDS("gridfits_d2.rds")

grid <- as.data.frame(grid_d2$d2_beta0.5_gaussian_rep1$Xg)
colnames(grid) <- c("x1", "x2")

f_ls <- grid_d2$d2_beta0.5_gaussian_rep4$f_ls
f_q <- grid_d2$d2_beta0.5_gaussian_rep4$f_q

f0 <- function(X) {
  X <- as.matrix(X)
  out <- rep(0, nrow(X))
  for (j in 1:25) {
    for (k in 1:25) {
      coeff <- (j * k)^(-(2 * 0.75 + 2/3))
      out <- out +
        coeff *
        sin(2 * pi * j * X[,1]) *
        sin(2 * pi * k * X[,2])
    }
  }
  out
}
f_true <- f0(grid)

df <- rbind(
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_true, method = "True"),
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_ls,   method = "LS"),
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_q,    method = "LAD")
)
df$method <- factor(df$method, levels = c("True", "LS", "LAD"))
z_range <- range(f_true)

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
ggplot(df, aes(x1, x2, fill = z)) +
  geom_tile() +
  facet_wrap(~ method, ncol = 3) +
  scale_fill_viridis_c(option = "C",
                       limits = z_range,
                       oob = scales::squish) +
  theme_minimal(base_size = 13) +
  labs(x = "", y = "", title = "", fill = "") +
  theme(
    panel.spacing = unit(0, "lines"),
    plot.margin = margin(0, 0, 0, 0),
    strip.text = element_blank(),
    legend.text = element_text(size = 20),
    axis.text = element_text(size = 20, colour = "black"),
    axis.ticks.length = unit(1.5, "mm"),
    legend.title = element_text(size = 20)
  ) + guides(fill = guide_colorbar(barwidth = 2, barheight = 12))
ggsave("Fig3.pdf", width = 20, height = 9, dpi = 320)

f_ls <- grid_d2$d2_beta0.5_t2_rep4$f_ls
f_q <- grid_d2$d2_beta0.5_t2_rep4$f_q

df <- rbind(
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_true, method = "True"),
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_ls,   method = "LS"),
  data.frame(x1 = grid$x1, x2 = grid$x2, z = f_q,    method = "LAD")
)

df$method <- factor(df$method, levels = c("True", "LS", "LAD"))
z_range <- range(f_true)

setwd("C:/Users/ik77w/OneDrive - University of Glasgow/Research/Robust RKHS")
ggplot(df, aes(x1, x2, fill = z)) +
  geom_tile() +
  # geom_raster(interpolate = TRUE) +
  # coord_fixed() + 
  # coord_cartesian(xlim = range(df$x1), ylim = range(df$x2)) + 
  facet_wrap(~ method, ncol = 3) +
  scale_fill_viridis_c(option = "C",
                       limits = z_range,
                       oob = scales::squish) +
  theme_minimal(base_size = 13) +
  labs(x = "", y = "", title = "", fill = "") +
  theme(
    panel.spacing = unit(0, "lines"),
    plot.margin = margin(0, 0, 0, 0),
    strip.text = element_blank(),
    legend.text = element_text(size = 20),
    axis.text = element_text(size = 20, colour = "black"),
    axis.ticks.length = unit(1.5, "mm"),
    legend.title = element_text(size = 20)
  ) + guides(fill = guide_colorbar(barwidth = 2, barheight = 12))
ggsave("Fig4.pdf", width = 20, height = 9, dpi = 320)


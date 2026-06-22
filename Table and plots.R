library(knitr)
library(kableExtra)
library(dplyr)
library(ggplot2)

# This script assumes that you have run the two simulation scripts and stored
# results_d1.rds and gridfits_d1.rds as well as results_d2.rds gridfits_d2.rds,
# see the simulation scripts for details.

####################### Table (Table 1) ########################################
################################################################################

# Combine results from the simulations
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

# Scale by 10 for readability

df <- df %>%
  mutate(
    mse_ls_mean = 10 * mse_ls_mean,
    mse_ls_sd   = 10 * mse_ls_sd,
    mse_q_mean  = 10 * mse_q_mean,
    mse_q_sd    = 10 * mse_q_sd,
    mse_h_mean  = 10 * mse_h_mean,
    mse_h_sd    = 10 * mse_h_sd
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
    mse_q_mean, mse_q_sd,
    mse_h_mean, mse_h_sd
  )


latex_out <- kable(
  tab,
  format = "latex",
  booktabs = TRUE,
  escape = FALSE,
  digits = 4,
  align = c("c","c","c","c","c","c","c", "c", "c"),
  col.names = c(
    "$d$",
    "$\\beta$",
    "Noise",
    "MSE", "SE",
    "MSE", "SE",
    "MSE", "SE"
  )
) %>%
  add_header_above(c(
    " " = 3,
    "LS" = 2,
    "LAD" = 2,
    "Huber" = 2
  )) %>%
  kable_styling(
    latex_options = c("hold_position", "scale_down"),
    full_width = FALSE,
    font_size = 10
  )

cat(latex_out)
# Copied and pasted into the paper, fixed the margins and lines

############################# Plots (Figures 1 - 4) ############################
################################################################################

# First one-dimensional estimates

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
  ) + guides(fill = guide_colorbar(barwidth = 2, barheight = 18))
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
  ) + guides(fill = guide_colorbar(barwidth = 2, barheight = 18))

ggsave("Fig4.pdf", width = 20, height = 9, dpi = 320)


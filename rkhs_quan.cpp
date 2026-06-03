// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_NO_DEBUG
#define ARMA_WARN_LEVEL 1

#include <RcppArmadillo.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace Rcpp;
using namespace arma;

// ============================================================
// Distance matrix
// ============================================================
static mat dist_matrix(const mat& X, const mat& Y) {
    uword n = X.n_rows, m = Y.n_rows;
    mat D(n, m);
    for (uword i = 0; i < n; ++i) {
        for (uword j = 0; j < m; ++j) {
            D(i, j) = norm(X.row(i) - Y.row(j), 2);
        }
    }
    return D;
}

// ============================================================
// Gaussian kernel
// ============================================================
// [[Rcpp::export]]
mat gaussian_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    return exp(-0.5 * square(D));
}

// ============================================================
// Matérn kernel
// ============================================================
// [[Rcpp::export]]
mat matern_kernel(const mat& X, const mat& Y, double m = 2.5, double ls = 1.0) {
    uword d = X.n_cols;
    double nu = m - 0.5 * d;

    if (X.n_cols != Y.n_cols) stop("X and Y must have the same number of columns");
    if (ls <= 0) stop("length_scale must be positive");
    if (nu <= 0) stop("m must be > d/2");

    mat D = dist_matrix(X, Y) / ls;

    if (std::abs(nu - 0.5) < 1e-8) return exp(-D);
    if (std::abs(nu - 1.5) < 1e-8) {
        mat T = sqrt(3.0) * D;
        return (1.0 + T) % exp(-T);
    }
    if (std::abs(nu - 2.5) < 1e-8) {
        mat T = sqrt(5.0) * D;
        return (1.0 + T + (T % T)/3.0) % exp(-T);
    }

    // General case
    mat K(D.n_rows, D.n_cols, fill::zeros);
    double a = sqrt(2.0 * nu);
    double c = pow(2.0, 1.0 - nu) / boost::math::tgamma(nu);

    for (uword i = 0; i < D.n_rows; ++i) {
        for (uword j = 0; j < D.n_cols; ++j) {
            double r = D(i, j);
            if (r < 1e-12) {
                K(i, j) = 1.0;
                continue;
            }
            double z = a * r;
            K(i, j) = c * pow(z, nu) * boost::math::cyl_bessel_k(nu, z);
        }
    }
    return K;
}

// ============================================================
// Smoothed quantile psi
// ============================================================
vec smoothed_quantile_psi(const vec& r, double tau, double eps = 1e-4) {
    vec psi = -tau * ones(r.n_elem);
    psi.elem(find(r > 0)) += 1.0;

    uvec near_zero = find(abs(r) < eps);
    if (!near_zero.is_empty()) {
        psi(near_zero) = (r(near_zero) / (2.0 * eps)) + (0.5 - tau);
    }
    return psi;
}

// ============================================================
// Main IRLS solver
// ============================================================
// [[Rcpp::export]]
List rkhs_quantile_irls(const mat& K, const vec& y,
                       double tau = 0.5,
                       double lambda = 1e-4,
                       double eps = 1e-4,
                       int max_iter = 100,
                       double tol = 1e-6) {

    uword n = K.n_rows;
    vec alpha(n, fill::zeros);
    int final_iter = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        final_iter = iter + 1;

        vec f = K * alpha;
        vec r = y - f;

        vec psi = smoothed_quantile_psi(r, tau, eps);
        vec w = abs(psi);
        w = w / (mean(w) + 1e-12);
        w = clamp(w, 1e-3, 5.0);

        mat W = diagmat(w);
        mat A = K * W * K + lambda * K;
        A.diag() += 1e-12;

        vec rhs = K * (w % y);

        vec alpha_new;
        bool success = solve(alpha_new, A, rhs, solve_opts::likely_sympd);

        if (!success) {
            warning("Solve failed at iteration %d", iter+1);
            break;
        }

        double diff = norm(alpha_new - alpha, 2);
        alpha = alpha_new;

        if (diff < tol) break;
    }

    return List::create(
        Named("alpha")      = alpha,
        Named("fitted")     = K * alpha,
        Named("iterations") = final_iter
    );
}

// ============================================================
// Fast prediction
// ============================================================
// [[Rcpp::export]]
vec rkhs_predict(const mat& X_train,
                 const mat& X_new,
                 const vec& alpha,
                 String kernel_type,
                 double ls = 1.0,
                 double m = 2.5) {
    
    mat K_new;
    if (kernel_type == "gaussian") {
        K_new = gaussian_kernel(X_train, X_new, ls);
    } else {
        K_new = matern_kernel(X_train, X_new, m, ls);
    }
    
    return trans(K_new) * alpha;   // n_new x n_train  *  n_train x 1
}
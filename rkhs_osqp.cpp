// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace Rcpp;
using namespace arma;


// ============================================================
// FAST Euclidean distance matrix (much faster than row tricks)
// ============================================================
static arma::mat dist_matrix(const arma::mat& X, const arma::mat& Y) {
    const arma::uword n = X.n_rows;
    const arma::uword m = Y.n_rows;

    arma::mat D(n, m, fill::zeros);

    for (arma::uword i = 0; i < n; ++i) {
        for (arma::uword j = 0; j < m; ++j) {
            D(i, j) = norm(X.row(i) - Y.row(j), 2);
        }
    }
    return D;
}


// ============================================================
// Gaussian kernel
// ============================================================
arma::mat gaussian_kernel(const arma::mat& X,
                          const arma::mat& Y,
                          double ls = 1.0) {

    arma::mat D = dist_matrix(X, Y) / ls;
    return arma::exp(-0.5 * arma::square(D));
}


// ============================================================
// Matérn kernel (Sobolev-parametrised)
// m > d/2, ν = m - d/2
// ============================================================
arma::mat matern_kernel(const arma::mat& X,
                        const arma::mat& Y,
                        double m,
                        double ls = 1.0) {

    const arma::uword d = X.n_cols;
    const double nu = m - 0.5 * d;

    if (X.n_cols != Y.n_cols)
        Rcpp::stop("X and Y must have same dimension.");

    if (ls <= 0)
        Rcpp::stop("ls must be positive.");

    if (nu <= 0)
        Rcpp::stop("Need m > d/2.");

    arma::mat D = dist_matrix(X, Y) / ls;

    // -----------------------------
    // Half-integer shortcuts
    // -----------------------------
    if (std::abs(nu - 0.5) < 1e-12)
        return arma::exp(-D);

    if (std::abs(nu - 1.5) < 1e-12) {
        arma::mat T = std::sqrt(3.0) * D;
        return (1.0 + T) % arma::exp(-T);
    }

    if (std::abs(nu - 2.5) < 1e-12) {
        arma::mat T = std::sqrt(5.0) * D;
        return (1.0 + T + (T % T) / 3.0) % arma::exp(-T);
    }

    // -----------------------------
    // General Bessel form
    // -----------------------------
    arma::mat K(D.n_rows, D.n_cols);

    const double a = std::sqrt(2.0 * nu);
    const double c = std::pow(2.0, 1.0 - nu) / boost::math::tgamma(nu);

    for (arma::uword i = 0; i < D.n_rows; ++i) {
        for (arma::uword j = 0; j < D.n_cols; ++j) {

            const double r = D(i, j);

            if (r < 1e-12) {
                K(i, j) = 1.0;
                continue;
            }

            const double z = a * r;

            K(i, j) =
                c *
                std::pow(z, nu) *
                boost::math::cyl_bessel_k(nu, z);
        }
    }

    return K;
}


// ============================================================
// Smoothed quantile subgradient
// ============================================================
arma::vec smoothed_quantile_psi(const arma::vec& r,
                                 double tau,
                                 double eps = 1e-5) {

    arma::vec psi = -tau * arma::ones(r.n_elem);
    psi.elem(arma::find(r > 0)).fill(1.0 - tau);

    arma::uvec idx = arma::find(arma::abs(r) < eps);

    if (!idx.is_empty()) {
        psi(idx) = (r(idx) / (2.0 * eps)) + (0.5 - tau);
    }

    return psi;
}


// ============================================================
// IRLS solver in RKHS
// ============================================================
Rcpp::List rkhs_quantile_irls(const arma::mat& K,
                              const arma::vec& y,
                              double tau = 0.5,
                              double lambda = 1e-4,
                              double eps = 1e-4,
                              int max_iter = 50,
                              double tol = 1e-6) {

    const arma::uword n = K.n_rows;

    arma::vec alpha(n, fill::zeros);

    arma::mat A;   // system matrix reused

    int iter = 0;

    for (; iter < max_iter; ++iter) {

        arma::vec f = K * alpha;
        arma::vec r = y - f;

        arma::vec psi = smoothed_quantile_psi(r, tau, eps);

        arma::vec w = arma::abs(psi);
        w = arma::clamp(w, 1e-8, arma::datum::inf);

        // -----------------------------
        // Stable weighted system:
        // K W K + n λ K
        // -----------------------------
        arma::mat KW = K.each_row() % w.t();  // K * W efficiently

        A = KW * K + (n * lambda) * K;

        arma::vec rhs = K * (w % y);

        arma::vec alpha_new;

        bool ok = arma::solve(alpha_new, A, rhs, arma::solve_opts::likely_sympd);

        if (!ok)
            break;

        if (arma::norm(alpha_new - alpha, 2) < tol) {
            alpha = alpha_new;
            break;
        }

        alpha = alpha_new;
    }

    return Rcpp::List::create(
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("iterations") = iter + 1
    );
}
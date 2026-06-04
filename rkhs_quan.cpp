// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_NO_DEBUG

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// ============================================================
// Distance
// ============================================================
static mat dist_matrix(const mat& X, const mat& Y) {
    uword n = X.n_rows, m = Y.n_rows;
    mat D(n, m);

    for (uword i = 0; i < n; ++i)
        for (uword j = 0; j < m; ++j)
            D(i,j) = norm(X.row(i) - Y.row(j), 2);

    return D;
}

// ============================================================
// Kernels
// ============================================================
mat gaussian_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    return exp(-0.5 * square(D));
}

mat matern_kernel(const mat& X, const mat& Y,
                  double s = 2.5, double ls = 1.0) {

    uword d = X.n_cols;
    double nu = s - 0.5 * d;

    if (nu <= 0)
        stop("Sobolev order s must satisfy s > d/2");

    mat D = dist_matrix(X, Y) / ls;

    if (std::abs(nu - 0.5) < 1e-8)
        return exp(-D);

    if (std::abs(nu - 1.5) < 1e-8) {
        mat T = sqrt(3.0) * D;
        return (1.0 + T) % exp(-T);
    }

    if (std::abs(nu - 2.5) < 1e-8) {
        mat T = sqrt(5.0) * D;
        return (1.0 + T + (T % T)/3.0) % exp(-T);
    }

    return exp(-D);
}

mat spherical_kernel(const mat& X, const mat& Y) {
    uword n = X.n_rows, m = Y.n_rows;
    mat K(n,m);

    for (uword i=0;i<n;i++)
        for (uword j=0;j<m;j++)
            K(i,j) = 1.0 / std::max(2.0 - dot(X.row(i), Y.row(j)), 1e-8);

    return K;
}

// ============================================================
// Kernel dispatcher
// ============================================================
// [[Rcpp::export]]
mat kernel_mat(const mat& X, const mat& Y,
               std::string type,
               double ls = 1.0,
               double s = 2.5) {

    if (type == "gaussian")
        return gaussian_kernel(X, Y, ls);

    if (type == "matern")
        return matern_kernel(X, Y, s, ls);

    return spherical_kernel(X, Y);
}

// ============================================================
// LS
// ============================================================
// [[Rcpp::export]]
List rkhs_ls(const mat& K, const vec& y, double lambda) {

    uword n = K.n_rows;
    mat A = K + n * lambda * eye(n,n);
    A.diag() += 1e-10;

    vec alpha = solve(A, y);

    return List::create(
        Named("alpha") = alpha,
        Named("fitted") = K * alpha
    );
}

// ============================================================
// LS OCV
// ============================================================
// [[Rcpp::export]]
double rkhs_ls_ocv(const mat& K, const vec& y, double lambda) {

    uword n = K.n_rows;

    mat A = K + n * lambda * eye(n,n);
    A.diag() += 1e-10;

    vec alpha = solve(A, y);
    vec f = K * alpha;
    vec r = y - f;

    mat H = K * solve(A, eye(n,n));
    vec h = H.diag();

    return mean(square(r / (1.0 - h + 1e-12)));
}

// ============================================================
// Nychka-style quantile smoothing IRLS (CLEAN VERSION)
// ============================================================
// [[Rcpp::export]]
List rkhs_quantile_irls(const mat& K,
                        const vec& y,
                        double tau = 0.5,
                        double lambda = 1e-4,
                        double eps = 1e-4,
                        int max_iter = 500,
                        double tol = 1e-6) {

    uword n = K.n_rows;

    vec alpha(n, fill::zeros);

    vec w(n);
    vec f, r;

    int it = 0;

    for (int k = 0; k < max_iter; k++) {

        it = k + 1;

        f = K * alpha;
        r = y - f;

        // ========================================================
        // NYCHKA QUADRATIC SMOOTHING (THIS IS THE KEY FIX)
        // ========================================================
        for (uword i = 0; i < n; i++) {

            if (r(i) > eps) {
                w(i) = 0.0;
            }
            else if (r(i) >= 0.0) {
                w(i) = 2.0 * tau / eps;
            }
            else if (r(i) > -eps) {
                w(i) = 2.0 * (1.0 - tau) / eps;
            }
            else {
                w(i) = 0.0;
            }
        }

        // numerical stability
        w += 1e-8;

        mat W = diagmat(w);

        mat A = K * W * K + lambda * eye(n, n);
        A.diag() += 1e-10;

        vec rhs = K * (w % y);

        vec alpha_new;

        bool ok = solve(alpha_new, A, rhs);
        if (!ok) break;

        if (norm(alpha_new - alpha, 2) < tol) {
            alpha = alpha_new;
            break;
        }

        alpha = alpha_new;
    }

    return List::create(
        Named("alpha") = alpha,
        Named("fitted") = K * alpha,
        Named("weights") = w,
        Named("iterations") = it
    );
}

// ============================================================
// Quantile GCV (unchanged, still valid for your surrogate)
// ============================================================
// [[Rcpp::export]]
double rkhs_quantile_gcv(const mat& K,
                          const vec& y,
                          const vec& alpha,
                          const vec& w,
                          double lambda) {

    uword n = K.n_rows;

    vec f = K * alpha;
    vec r = y - f;

    mat W = diagmat(w);
    mat A = K * W * K + lambda * eye(n, n);
    A.diag() += 1e-10;

    mat H = K * solve(A, K * W);
    vec h = H.diag();

    double out = 0.0;

    for (uword i = 0; i < n; i++) {
        double d = 1.0 - h(i);
        out += w(i) * r(i) * r(i) / (d * d + 1e-12);
    }

    return out / n;
}

// ============================================================
// Prediction
// ============================================================
// [[Rcpp::export]]
vec rkhs_predict(const mat& Xtr,
                 const mat& Xte,
                 const vec& alpha,
                 std::string kernel,
                 double ls = 1.0,
                 double s = 2.5) {

    mat K = kernel_mat(Xtr, Xte, kernel, ls, s);
    return K.t() * alpha;
}
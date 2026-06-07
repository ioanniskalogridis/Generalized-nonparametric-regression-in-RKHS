// [[Rcpp::depends(RcppArmadillo)]]
#define ARMA_NO_DEBUG
#include <RcppArmadillo.h>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>


using namespace Rcpp;
using namespace arma;

// Distance
static mat dist_matrix(const mat& X, const mat& Y) {
    uword n = X.n_rows, m = Y.n_rows;
    mat D(n, m);
    
    // Faster vectorized version
    for (uword i = 0; i < n; ++i) {
        D.row(i) = sqrt(sum(pow(X.row(i) - Y.each_row(), 2), 1)).t();
    }
    return D;
}

// Gaussian kernel
mat gaussian_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    return exp(-0.5 * square(D));
}

// Matérn (Sobolev) kernel (1D interpretation only)
// s = smoothness parameter

mat matern_kernel(const mat& X,
                  const mat& Y,
                  double s = 2.5,
                  double ls = 1.0) {

    mat D = dist_matrix(X, Y) / ls;

    mat K(D.n_rows, D.n_cols, fill::zeros);

    double nu = s;

    double c1 = std::pow(2.0, 1.0 - nu) / std::tgamma(nu);

    for (uword i = 0; i < D.n_rows; i++) {
        for (uword j = 0; j < D.n_cols; j++) {

            double r = std::sqrt(2.0 * nu) * D(i,j);

            if (r < 1e-12) {
                K(i,j) = 1.0;
            } else {
                double b = boost::math::cyl_bessel_k(nu, r);
                K(i,j) = c1 * std::pow(r, nu) * b;
            }
        }
    }

    return K;
}

// Tensor Sobolev kernel (arbitrary dimension d)
// K(x,y) = Π_j K_s(x_j, y_j) - product kernel

// [[Rcpp::export]]
mat tensor_sobolev_kernel(const mat& X, const mat& Y, double ls, double s) {
    uword d = X.n_cols;
    mat K(X.n_rows, Y.n_rows, fill::ones);
    
    for (uword j = 0; j < d; ++j) {
        mat Xj = X.col(j);
        mat Yj = Y.col(j);
        mat Kj = matern_kernel(Xj, Yj, s, ls);   // reuse fast matern
        K = K % Kj;
    }
    return K;
}
// Dispatcher

// [[Rcpp::export]]
mat kernel_mat(const mat& X, const mat& Y, std::string type, double ls = 1.0, double s = 2.5) {
    if (type == "gaussian") return gaussian_kernel(X, Y, ls);
    if (type == "matern")   return matern_kernel(X, Y, s, ls);
    if (type == "tensor")   return tensor_sobolev_kernel(X, Y, ls, s);
    stop("Unknown kernel");
}

// LS estimator

// [[Rcpp::export]]
List rkhs_ls(const mat& K, const vec& y, double lambda) {

    uword n = K.n_rows;

    mat A = K + n * lambda * eye(n,n);
    // A.diag() += 1e-12; // Just in case, but overall not necessary

    vec alpha = solve(A, y);

    return List::create(
        Named("alpha") = alpha,
        Named("fitted") = K * alpha
    );
}

// Leave-one-out CV (LS)

// [[Rcpp::export]]
double rkhs_ls_ocv(const mat& K, const vec& y, double lambda) {

    uword n = K.n_rows;

    mat A = K + n * lambda * eye(n,n);

    vec alpha = solve(A, y);
    vec f = K * alpha;
    vec r = y - f;

    mat H = K * solve(A, eye(n,n));
    vec h = H.diag();

    return mean(square(r / (1.0 - h )));
}

// Quantile IRLS with local quadratic approximation

// [[Rcpp::export]]
List rkhs_quantile_irls(const mat& K,
                        const vec& y,
                        double tau = 0.5,
                        double lambda = 1e-4,
                        int max_iter = 100,
                        double eps = 1e-03,
                        double tol = 1e-8) {

uword n = K.n_rows;

vec alpha(n, fill::zeros);
vec w(n, fill::ones);

for (int k = 0; k < max_iter; k++) {

    vec f = K * alpha;
    vec r = y - f;

    for (uword i = 0; i < n; i++) {

        // Huberisation inside quadratic band 
        if (std::abs(r(i)) <= eps) {

            if (r(i) >= 0)
                w(i) = tau / eps;
            else
                w(i) = (1.0 - tau) / eps;

        } 
        // Regular IRLS weights outside
        else {

            if (r(i) > 0)
                w(i) = tau/r(i);
            else
                w(i) = (tau - 1) /r(i);
        }
    }
    
        // Rcout << sum(w > 0) << std::endl; // just to check

        mat W = diagmat(w);

        mat A = W * K + 2*n*lambda* eye(n,n);
        vec rhs = (w % y);
        //vec rhs = K*(w % y);

        vec alpha_new;

        if (!solve(alpha_new, A, rhs))
            break;

        if (norm(alpha_new - alpha, 2) < tol) {
            alpha = alpha_new;
            break;
        }

        alpha = alpha_new;
    }

    return List::create(
        Named("alpha") = alpha,
        Named("fitted") = K * alpha,
        Named("weights") = w
    );
}

// Quantile robust OCV

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
        mat A = W * K + 2*n*lambda* eye(n,n); 
        mat H = K * solve(A, W); 
        vec h = H.diag(); 
        // double out = 0.0; 
        //for (uword i = 0; i < n; i++) { 
        //    double d = 1.0 - h(i); out += w(i) * r(i) * r(i) / (d * d); 
        //} 
        //return out / n; 
        return mean(w%square(r / (1.0 - h )));
    }

// Prediction

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
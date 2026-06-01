// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// =============================================
// Helper: Euclidean distance matrix
// =============================================
mat dist_matrix(const mat& X, const mat& Y) {
    int n = X.n_rows, m = Y.n_rows;
    mat D(n, m);
    for(int i = 0; i < n; i++) {
        D.row(i) = sqrt(sum(pow(X.row(i) - Y.each_row(), 2), 1)).t();
    }
    return D;
}

// =============================================
// Kernels
// =============================================
// [[Rcpp::export]]
mat gaussian_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    return exp(-0.5 * pow(dist_matrix(X, Y)/ls, 2));
}

// [[Rcpp::export]]
mat matern52_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    mat tmp = sqrt(5.0) * D;
    return (1.0 + tmp + (tmp % tmp)/3.0) % exp(-tmp);
}

// =============================================
// Smoothed subgradient of quantile (pinball) loss
// =============================================
vec smoothed_quantile_psi(const vec& r, double tau, double epsilon = 1e-5) {
    vec psi = -tau * ones(r.n_elem);      // base for negative residuals
    psi.elem(find(r > 0)) += 1.0;         // adjust for positive residuals
    
    // Smooth approximation around zero
    uvec near_zero = find(abs(r) < epsilon);
    if (near_zero.n_elem > 0) {
        psi(near_zero) = (r(near_zero) / (2.0 * epsilon)) + (0.5 - tau);
    }
    return psi;
}

// =============================================
// Main IRLS solver for smoothed quantile loss
// =============================================
// [[Rcpp::export]]
List rkhs_quantile_irls(const mat& K, const vec& y, 
                       double tau = 0.5, double lambda = 1e-4,
                       double epsilon = 1e-4, int max_iter = 100, double tol = 1e-6) {
    
    int n = K.n_rows;
    vec alpha = zeros(n);           // coefficients in RKHS (representer theorem)
    int final_iter = 0;
    
    for(int iter = 0; iter < max_iter; iter++) {
        final_iter = iter + 1;
        
        vec f = K * alpha;                    // current predictions
        vec r = y - f;                        // residuals
        
        // Compute smoothed subgradient and weights
        vec psi = smoothed_quantile_psi(r, tau, epsilon);
        vec w = abs(psi);
        w.elem(find(w < 1e-8)).fill(1e-8);    // prevent zero weights
        
        // Weighted least squares system
        mat W = diagmat(w);
        mat lhs = K * W * K + (n * lambda) * K;   // no extra ridge
        vec rhs = K * (w % y);
        
        vec alpha_new;
        try {
            alpha_new = solve(lhs, rhs, solve_opts::likely_sympd);
        } catch (...) {
            alpha_new = alpha;   // fallback on failure
        }
        
        double diff = norm(alpha_new - alpha, 2);
        alpha = alpha_new;
        
        if (diff < tol) break;
    }
    
    return List::create(Named("alpha") = alpha,
                        Named("iterations") = final_iter);
}
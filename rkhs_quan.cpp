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
    
    // Much faster vectorized version
    for (uword i = 0; i < n; ++i) {
        D.row(i) = sqrt(sum(pow(X.row(i) - Y.each_row(), 2), 1)).t();
    }
    return D;
}

// ============================================================
// Gaussian kernel
// ============================================================
mat gaussian_kernel(const mat& X, const mat& Y, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    return exp(-0.5 * square(D));
}

// ============================================================
// Matérn (Sobolev) kernel (1D interpretation only)
// s = smoothness parameter (do NOT adjust for dimension here)
// ============================================================
mat matern_kernel(const mat& X, const mat& Y, double s = 2.5, double ls = 1.0) {
    mat D = dist_matrix(X, Y) / ls;
    
    if (std::abs(s - 2.5) < 1e-8) {
        mat T = sqrt(5.0) * D;
        return (1.0 + T + (T % T)/3.0) % exp(-T);
    }
    if (std::abs(s - 1.5) < 1e-8) {
        mat T = sqrt(3.0) * D;
        return (1.0 + T) % exp(-T);
    }
    if (std::abs(s - 0.5) < 1e-8) return exp(-D);
    
    return exp(-D); // fallback
}

// ============================================================
// Tensor Sobolev kernel (arbitrary dimension d)
// K(x,y) = Π_j K_s(x_j, y_j)
// ============================================================
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
// ============================================================
// Dispatcher
// ============================================================
// [[Rcpp::export]]
mat kernel_mat(const mat& X, const mat& Y, std::string type, double ls = 1.0, double s = 2.5) {
    if (type == "gaussian") return gaussian_kernel(X, Y, ls);
    if (type == "matern")   return matern_kernel(X, Y, s, ls);
    if (type == "tensor")   return tensor_sobolev_kernel(X, Y, ls, s);
    stop("Unknown kernel");
}

// ============================================================
// Ridge / LS
// ============================================================
// [[Rcpp::export]]
List rkhs_ls(const mat& K, const vec& y, double lambda) {

    uword n = K.n_rows;

    mat A = K + n * lambda * eye(n,n);
    // A.diag() += 1e-12;

    vec alpha = solve(A, y);

    return List::create(
        Named("alpha") = alpha,
        Named("fitted") = K * alpha
    );
}

// ============================================================
// Leave-one-out CV (LS)
// ============================================================
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

// ============================================================
// Nychka-style quantile IRLS
// ============================================================
// [[Rcpp::export]]
List rkhs_quantile_irls(const mat& K,
                        const vec& y,
                        double tau = 0.5,
                        double lambda = 1e-4,
                        int max_iter = 100,
                        double eps = 1e-02,
                        double tol = 1e-8) {

// ouble eps = 0.1 * arma::stddev(y);
// if (eps < 1e-10) eps = 1e-10;

uword n = K.n_rows;

vec alpha(n, fill::zeros);
vec w(n, fill::ones);

for (int k = 0; k < max_iter; k++) {

    vec f = K * alpha;
    vec r = y - f;

    for (uword i = 0; i < n; i++) {

        // inside quadratic band (Huberisation)
        if (std::abs(r(i)) <= eps) {

            if (r(i) >= 0)
                w(i) = tau / eps;
            else
                w(i) = (1.0 - tau) / eps;

        } 
        // outside band: true quantile IRLS form (ψ/r)
        else {

            if (r(i) > 0)
                w(i) = tau/r(i);
            else
                w(i) = (tau - 1) /r(i);
        }
    }
    
        // Rcout << sum(w > 0) << std::endl;

        mat W = diagmat(w);

        mat A = W * K + 2*n*lambda* eye(n,n);
        //mat A = K*W * K + 2*n*lambda*K;
        A.diag() += 1e-12;
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
        // mat A = K* W * K + 2*n*lambda*K; 
        A.diag() += 1e-12;
        mat H = K * solve(A, W); 
        // mat H = K * solve(A, K*W); 
        vec h = H.diag(); 
        double out = 0.0; 
        for (uword i = 0; i < n; i++) { 
            double d = 1.0 - h(i); out += w(i) * r(i) * r(i) / (d * d); 
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
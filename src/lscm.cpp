#include "lscm.h"
#include <igl/cotmatrix.h>
#include "vector_area_matrix.h"
#include <igl/repdiag.h>
#include <igl/massmatrix.h>
#include <igl/eigs.h>
#include <Eigen/SVD>

void lscm(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  // Replace with your code

  // Define Q = [(L,0), (0, L)] - A
  Eigen::SparseMatrix<double> L, A, Q;
  igl::cotmatrix(V, F, L); 
  vector_area_matrix(F, A);
  igl::repdiag(L, 2, Q);
  Q -= A;

  // Define B = [(M,0), (0, M)]
  Eigen::SparseMatrix<double> M, B;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
  igl::repdiag(M, 2, B);

  //Solve Generalized Eigenvalue problem
  //sU  #A by k list of sorted eigen vectors (descending)
  Eigen::MatrixXd sU;
  //sS  k list of sorted eigen values (descending)
  Eigen::VectorXd sS;
  // We need the first 3 since the first 2 map to the solution with zero energy (eigenvalue of zero)
  igl::eigs(Q, B, 3, igl::EIGS_TYPE_SM, sU, sS);

  // Canonical Rotation
  U.resize(V.rows(), 2);
  // Selects the first n rows of the second column
  U.col(0) = sU.col(2).head(V.rows());
  // Selects the last n rows of the second column
  U.col(1) = sU.col(2).tail(V.rows());

  // Get svd of U.transpose() * U
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose()*U, Eigen::ComputeThinU | Eigen::ComputeThinV);
  U = U*svd.matrixU();
}

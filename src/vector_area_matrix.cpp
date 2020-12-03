#include "vector_area_matrix.h"
#include <igl/boundary_loop.h>
#include <vector>
void vector_area_matrix(
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double>& A)
{ 
  // Define A
  int n = F.maxCoeff() + 1;
  A.resize(2 * n, 2 * n);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(n);

  //LL is list of loops where LL[i] = ordered list of boundary vertices in loop i
  std::vector<std::vector<int>> LL;
  igl::boundary_loop(F, LL);

  for(int i = 0; i < LL.size(); i++){
    for (int j = 0; j < LL[i].size(); j++){
      int ui = LL[i][j];
			int uj = LL[i][(j+1) % LL[i].size()];

      tripletList.push_back(T(ui, uj+n, 1));
      tripletList.push_back(T(ui+n, uj, -1));
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  // Do new A = 0.5*(A+A.transpose) to get a symmetric matrix
  A += Eigen::SparseMatrix<double>(A.transpose());
  A *= 0.5;
}


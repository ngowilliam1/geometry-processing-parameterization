#include "tutte.h"
#include <igl/edges.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>

void tutte(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  // Replace with your code
  // Calculate graph laplacian by first getting matrix of edges
  Eigen::MatrixXi edges;
	igl::edges(F, edges);
  Eigen::SparseMatrix<double> L;

  // laplacian for index i != j
	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(edges.rows() * 2);
	for (int edge = 0; edge < edges.rows(); edge++) {
    int ei = edges(edge, 0);
    int ej = edges(edge, 1);
		tripletList.push_back(T(ei, ej, 1.0));
		tripletList.push_back(T(ej, ei, 1.0));
	}
	L.resize(edges.maxCoeff() + 1, edges.maxCoeff() + 1);
	L.setFromTriplets(tripletList.begin(), tripletList.end());

	// laplacian on the diagonal i=j
	for (int ij = 0; ij < (edges.maxCoeff() + 1); ij++) {
		L.insert(ij, ij) = -L.row(ij).sum();
	}

  // Compute ordered boundary loops for a manifold mesh and return the longest loop in terms of vertices.
  Eigen::VectorXi BL;
  igl::boundary_loop(F, BL);

  // Map the vertices whose indices are in a given boundary loop (bnd) on the unit circle
  Eigen::MatrixXd UV; 
  igl::map_vertices_to_circle(V, BL ,UV);

  // Minimize Energy with boundary vertices mapped to unit circle
  igl::min_quad_with_fixed_data<double> data;
  Eigen::SparseMatrix<double> Aeq;
  Eigen::MatrixXd Beq, B;
  B = Eigen::MatrixXd::Zero(V.rows(), 2);
  igl::min_quad_with_fixed_precompute(L, BL, Aeq, false, data);
  igl::min_quad_with_fixed_solve(data, B, UV, Beq, U);
}


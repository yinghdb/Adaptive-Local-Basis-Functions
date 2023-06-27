#include <torch/extension.h>
// #include "multi_ball_query/multi_ball_query.h"
#include "furthest_point_sampling/furthest_point_sampling.h"
#include "compute_rbf/compute_rbf.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("multi_ball_query", &multi_ball_query, "multi_ball_query");
  m.def("furthest_point_sampling", &furthest_point_sampling, "furthest_point_sampling");
  m.def("compute_rbf", &compute_rbf, "compute_rbf");
}

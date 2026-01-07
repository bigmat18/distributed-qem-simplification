#include "logging.hpp"
#include "mesh_import.hpp"
#include <cstdint>
#include <cstdio>
#include <cxxopts.hpp>
#include <mpi.h>
#include <omp.h>
#include <qem_mesh.hpp>
#include <qem_simp.hpp>
#include <uniform_grid.hpp>
#include <utils.hpp>

int main(int argc, char *argv[]) {
  cxxopts::Options options("cli",
                           "CLI app to test distributed mesh simplification");
  options.add_options()("i,filename", "Input filename list",
                        cxxopts::value<std::string>())(
      "n,target", "Target faces", cxxopts::value<uint32_t>());

  options.parse_positional({"filename"});
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    printf("%s", options.help().c_str());
    return 0;
  }

  massert(result.count("filename") >= 1, "Need [input filename]");
  const std::string FILENAME = result["filename"].as<std::string>();
  const uint32_t TARGET_FACES = result["target"].as<uint32_t>();

  qems::MeshData data;
  qems::import_mesh<qems::ImportType::ROW_MESH_DATA>(FILENAME, data);
  for (const auto el : data.row_vertices)
    LOG_INFO("{}", el);
  // char message[20];
  // int rank;

  // MPI_Init(&argc, &argv);
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // if (rank == 0) {

  // qems::QEMMesh mesh;
  // massert(OpenMesh::IO::read_mesh(mesh, FILENAME), "Error in mesh import");

  // LOG_INFO("{} successfully imported", FILENAME.c_str());
  // mesh.request_vertex_status();
  // mesh.request_edge_status();
  // mesh.request_face_status();
  // mesh.request_halfedge_status();

  // qems::UniformGrid uniform_grid;
  //{
  // PROFILING_SCOPE("Pre-Processing");
  // Eigen::Vector3d min;
  // Eigen::Vector3d max;

  // qems::compute_bounding_box(mesh, min, max);

  // auto set_neighborhood = [&](qems::QEMMesh::VertexHandle vh) {
  // if (!mesh.data(vh).Collasable)
  // return;

  // for (auto vvh : mesh.vv_range(vh))
  // mesh.data(vvh).Collasable = false;
  //};

  // LOG_DEBUG("Start UniformGrid building");
  // for (size_t i = 0; i < mesh.n_vertices(); ++i) {
  // auto vh = qems::QEMMesh::VertexHandle(i);
  // mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
  // uint32_t idx = uniform_grid.add_vertex(mesh, vh);
  // mesh.data(vh).NodeIdx = idx;
  //}

  // for (size_t i = 0; i < mesh.n_edges(); ++i) {
  // auto eh = qems::QEMMesh::EdgeHandle(i);
  // auto heh = mesh.halfedge_handle(eh, 0);
  // auto vh0 = mesh.from_vertex_handle(heh);
  // auto vh1 = mesh.to_vertex_handle(heh);

  // uint32_t idx0 = mesh.data(vh0).NodeIdx;
  // uint32_t idx1 = mesh.data(vh1).NodeIdx;

  // if (idx0 != idx1) {
  // mesh.data(vh0).Collasable = false;
  // mesh.data(vh1).Collasable = false;

  // set_neighborhood(vh0);
  // set_neighborhood(vh1);
  //}
  //}

  // for (size_t i = 0; i < mesh.n_edges(); ++i) {
  // auto eh = qems::QEMMesh::EdgeHandle(i);

  // if(uniform_grid.add_edge(mesh, eh)) {
  // auto heh = mesh.halfedge_handle(eh, 0);
  // auto vh0 = mesh.from_vertex_handle(heh);
  // auto vh1 = mesh.to_vertex_handle(heh);

  // Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
  // Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

  // mesh.data(eh).Error = newV.transpose() * Q * newV;
  // mesh.data(eh).NewVertex = newV;
  //}
  //}

  // for(size_t i = 0; i < mesh.n_faces(); i++) {
  // auto fh = qems::QEMMesh::FaceHandle(i);
  // uniform_grid.increment_collasable_faces(mesh, fh);
  //}
  //}

  //} else {
  //}

  // MPI_Finalize();
  return 0;
}

#include <cstdint>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>

#include <mpi.h>
#include <utils.hpp>

#include "profiling.hpp"
#include "qem_mesh.hpp"
#include "qem_simp.hpp"
#include "uniform_grid.hpp"
#include "mesh_import.hpp"
#include "async_send.hpp"
#include "sync_send_recv.hpp"
#include "message_layout.hpp"
#include "packed_message.hpp"

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4
#define CSTM_MESH 5

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,target", "Target faces", cxxopts::value<uint32_t>());

    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    const std::string INPUT           = result["input"].as<std::string>();
    const uint32_t    TARGET_FACES    = result["target"].as<uint32_t>();

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    mpi::MessageLayout layout(CSTM_MESH);
    layout 
     .add_buffer<double>(CSTM_TAG_BB)
     .add_buffer<float>(CSTM_TAG_VERT)
     .add_buffer<uint32_t>(CSTM_TAG_FACE);

    if (pid == 0) {
        std::vector<fs::path> files;
        for (const auto& file : fs::directory_iterator(INPUT))
        if (fs::is_regular_file(file.status()))
            files.push_back(file);

        uint32_t current_file_idx = 0;
        uint32_t active_workers = 0;
        mpi::UnpackedAsyncSend send(layout);
        qems::MeshData load_data;

        for (int dest = 1; dest < num_procs; ++dest) {
            if (current_file_idx >= files.size()) 
                break;  

            const auto& file = files[current_file_idx];
            const std::string name = file.filename().string();
            {
                PROFILING_SCOPE("Sending-" + name);

                qems::import_mesh(file, load_data);
                LOG_INFO("{} - Imported {} with {} vertices, {} faces", 
                         pid, name,
                         load_data.row_vertices.size() / 3, 
                         load_data.row_faces.size() / 3);

                const auto& min = load_data.min_coords;
                const auto& max = load_data.max_coords;

                std::vector<double> bb(6);
                bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                send.isend(dest, {
                    {CSTM_TAG_BB, std::move(bb)},
                    {CSTM_TAG_VERT, std::move(load_data.row_vertices)},
                    {CSTM_TAG_FACE, std::move(load_data.row_faces)}
                });

                current_file_idx++;
                active_workers++;
            }
            PROFILING_PRINT();
        }

        mpi::PackedMessage end_msg;
        mpi::PackedMessage msg(layout);
        const auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
        const auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
        const auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

        while(active_workers > 0) {
            int dest = mpi::unpacked_sync_recv(msg); 

            if (current_file_idx < files.size()) {
                const auto& file = files[current_file_idx];

                {
                    PROFILING_SCOPE("Sending-" + file.filename().string());

                    qems::import_mesh(file, load_data);
                    LOG_INFO("{} - Imported {} vertices, {} faces", 
                             pid,
                             load_data.row_vertices.size() / 3, 
                             load_data.row_faces.size() / 3);

                    const auto& min = load_data.min_coords;
                    const auto& max = load_data.max_coords;

                    std::vector<double> bb(6);
                    bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
                    bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

                    send.isend(dest, {
                        {CSTM_TAG_BB, std::move(bb)},
                        {CSTM_TAG_VERT, std::move(load_data.row_vertices)},
                        {CSTM_TAG_FACE, std::move(load_data.row_faces)}
                    });

                    current_file_idx++;
                }
               PROFILING_PRINT();

            } else {
                mpi::unpacked_sync_send(dest, end_msg);
                active_workers--;
            }

            //qems::QEMMesh mesh;
            //qems::row_data_to_mesh(vertices, faces, mesh);
            //std::string final_name(name.begin(), name.end());
            //massert(OpenMesh::IO::write_mesh(mesh, "out/" + final_name), "Error in mesh export!");
        }

    } else {
        mpi::PackedMessage input_msg(layout);
        mpi::PackedMessage output_msg(layout);
        auto& bb = input_msg.get_buffer<double>(CSTM_TAG_BB);
        auto& vertices = input_msg.get_buffer<float>(CSTM_TAG_VERT);
        auto& faces = input_msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

        Eigen::Vector3d min, max;

        while (true) {
            if (mpi::unpacked_sync_recv(input_msg, 0) == -1) 
                break;

            LOG_INFO("{} - Recived {} vertices, {} faces", 
                     pid,
                     vertices.size() / 3, 
                     faces.size() / 3);


            min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2];
            max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5];

            qems::QEMMesh mesh;
            qems::UniformGrid uniform_grid;

            mesh.request_vertex_status();
            mesh.request_edge_status();
            mesh.request_face_status();
            mesh.request_halfedge_status();

            //{
                //PROFILING_SCOPE("QEM-Sim-Rank-" + std::to_string(pid) + "-" + final_name);
                //{
                    //PROFILING_SCOPE("Mesh-Import");
                    //qems::row_data_to_mesh(vertices, faces, mesh);
                //}      

                //{
                    //PROFILING_SCOPE("Pre-Processing");

                    //uniform_grid = qems::UniformGrid(min, max, 8);
                    //#pragma omp declare reduction(                                      \
                        //uniform_grid_merge : qems::UniformGrid : omp_out.merge(omp_in)) \
                        //initializer(omp_priv = qems::UniformGrid(omp_orig)              \
                    //)

                    //auto set_neighborhood = [&](qems::QEMMesh::VertexHandle vh) {
                        //if (!mesh.data(vh).Collasable) 
                            //return;

                        //for (auto vvh : mesh.vv_range(vh))
                        //mesh.data(vvh).Collasable = false;
                    //};

                    //PROFILING_LOCK();
                    //#pragma omp parallel reduction(uniform_grid_merge : uniform_grid)
                    //{
                        //PROFILING_SCOPE("Uniform-Grid-Building");
                        //{
                            //#pragma omp for schedule(static) 
                            //for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                                //auto vh = qems::QEMMesh::VertexHandle(i);
                                //mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                                //uint32_t idx = uniform_grid.add_vertex(mesh, vh);
                                //mesh.data(vh).NodeIdx = idx;
                            //}


                            //#pragma omp for schedule(static)  
                            //for (size_t i = 0; i < mesh.n_edges(); ++i) {
                                //auto eh = qems::QEMMesh::EdgeHandle(i);
                                //auto heh = mesh.halfedge_handle(eh, 0);
                                //auto vh0 = mesh.from_vertex_handle(heh);
                                //auto vh1 = mesh.to_vertex_handle(heh);

                                //uint32_t idx0 = mesh.data(vh0).NodeIdx;
                                //uint32_t idx1 = mesh.data(vh1).NodeIdx;

                                //if (idx0 != idx1) {
                                    //mesh.data(vh0).Collasable = false;
                                    //mesh.data(vh1).Collasable = false;

                                    //set_neighborhood(vh0);
                                    //set_neighborhood(vh1);
                                //} 
                            //}

                            //#pragma omp for schedule(static)
                            //for (size_t i = 0; i < mesh.n_edges(); ++i) {
                                //auto eh = qems::QEMMesh::EdgeHandle(i);

                                //if(uniform_grid.add_edge(mesh, eh)) {
                                    //auto heh = mesh.halfedge_handle(eh, 0);
                                    //auto vh0 = mesh.from_vertex_handle(heh);
                                    //auto vh1 = mesh.to_vertex_handle(heh);

                                    //Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                                    //Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                                    //mesh.data(eh).Error = newV.transpose() * Q * newV;
                                    //mesh.data(eh).NewVertex = newV;
                                //}
                            //}

                            //#pragma omp for schedule(static)
                            //for(size_t i = 0; i < mesh.n_faces(); i++) {
                                //auto fh = qems::QEMMesh::FaceHandle(i);
                                //uniform_grid.increment_collasable_faces(mesh, fh);
                            //}
                        //}

                    //}
                    //PROFILING_UNLOCK();
                //}

                //{
                    //PROFILING_SCOPE("Processing");

                    //#pragma omp parallel for schedule(dynamic, 1)
                    //for (const auto &cell : uniform_grid) {
                        //auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), cell.edges);

                        //uint32_t local_num_faces = cell.collasable_faces; 
                        //float total_faces = static_cast<float>(uniform_grid.total_collasable_faces());
                        //float cell_faces  = static_cast<float>(local_num_faces);

                        //float fraction = (total_faces > 0.0) ? (cell_faces / total_faces) : 0.0;
                        //float target_d = static_cast<float>(TARGET_FACES) * fraction;

                        //uint32_t local_target = static_cast<uint32_t>(std::floor(target_d));

                        //qems::simplification(mesh, local_target, local_num_faces, pq);
                    //}
                //}

                //{
                    //PROFILING_SCOPE("Mesh-Cleanup");
                    //mesh.garbage_collection();
                //}

                //{
                    //PROFILING_SCOPE("Refinements");
                    //std::vector<qems::QEMMesh::EdgeHandle> edges;
                    //edges.reserve(mesh.n_edges());

                    //#pragma omp parallel
                    //{
                        //std::vector<qems::QEMMesh::EdgeHandle> local_edges;
                        //size_t n = mesh.n_edges();
                        //int num_threads = omp_get_num_threads();
                        //local_edges.reserve((n + num_threads - 1) / num_threads);

                        //#pragma omp for schedule(static) 
                        //for (size_t i = 0; i < mesh.n_vertices(); ++i) {
                            //auto vh = qems::QEMMesh::VertexHandle(i);
                            //mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                        //}

                        //#pragma omp for schedule(static)
                        //for (size_t i = 0; i < mesh.n_edges(); ++i) {
                            //auto eh = qems::QEMMesh::EdgeHandle(i);
                            //auto heh = mesh.halfedge_handle(eh, 0);
                            //auto vh0 = mesh.from_vertex_handle(heh);
                            //auto vh1 = mesh.to_vertex_handle(heh);

                            //Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                            //Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                            //mesh.data(eh).Error = newV.transpose() * Q * newV;
                            //mesh.data(eh).NewVertex = newV;
                            //local_edges.push_back(eh);
                        //}

                        //#pragma omp critical
                        //{
                            //edges.insert(edges.end(),
                                         //local_edges.begin(),
                                         //local_edges.end());
                        //}
                    //}

                    //qems::QEMPriorityQueue pq(qems::QEMEdgeCompare(mesh), edges);
                    //qems::simplification(mesh, TARGET_FACES, mesh.n_faces(), pq);
                    //mesh.garbage_collection();
                //}
            //}
            //PROFILING_PRINT();
            //auto& out_verts = output_msg.get_buffer<float>(CSTM_TAG_VERT);
            //auto& out_faces = output_msg.get_buffer<uint32_t>(CSTM_TAG_FACE);

            //qems::mesh_to_row_data(mesh, out_verts, out_faces);
            mpi::unpacked_sync_send(0, output_msg);
        }
    }

    MPI_Finalize();
	return 0;
}


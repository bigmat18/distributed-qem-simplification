#include "logging.hpp"
#include "massert.hpp"
#include "packed_message.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <cxxopts.hpp>
#include <queue>

#include <mpi.h>
#include <utility>
#include <utils.hpp>

#include <ug_row_data.hpp>
#include <mesh_import.hpp>
#include <async_send.hpp>
#include <message_layout.hpp>
#include <sync_send_recv.hpp>
#include <qem_simp.hpp>
#include <qem_mesh.hpp>

#define CSTM_TAG_BB 1
#define CSTM_TAG_VERT 2
#define CSTM_TAG_FACE 3
#define CSTM_TAG_NAME 4

#define CSTM_TAG_CELL_ID 6
#define CSTM_TAG_CELL_PART_LVL 7 
#define CSTM_TAG_IDX_MAP 8
#define CSTM_TAG_FINAL_TARGET 9

#define CSTM_MESH 5

inline int next_step(int n) {
    if (n == 1) return 0;
    return (n % 2 != 0) ? (n - 1) / 2 : (n > 4 ? n / 2 + 1 : 1);
}

int main (int argc, char *argv[]) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("n,meshes", "Num meshes", cxxopts::value<uint32_t>())
        ("p,partitions", "Start partitions", cxxopts::value<uint32_t>()->default_value("4"));
 
    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);
 
    if(result.count("help")) { 
        printf("%s", options.help().c_str()); 
        return 0; 
    } 
 
    const std::string INPUT             = result["input"].as<std::string>();
    const uint32_t    NUM_MESHES        = result["meshes"].as<uint32_t>();
    const uint32_t    START_PARTITIONS  = result["partitions"].as<uint32_t>();

    massert(fs::exists("out") && fs::is_directory("out"), 
            "out folder does not exists");
    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");

	int pid, num_procs;
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
	MPI_Comm_rank(MPI_COMM_WORLD,&pid); 

    const float TARGET_FACES = 0.1;

    mpi::MessageLayout layout;
    layout
     .add_buffer<uint32_t, 2>(CSTM_TAG_CELL_ID)
     .add_buffer<uint32_t, 2>(CSTM_TAG_CELL_PART_LVL)
     .add_buffer<uint32_t, 1>(CSTM_TAG_FINAL_TARGET)
     .add_buffer<double, 6>(CSTM_TAG_BB)
     .add_buffer<char>(CSTM_TAG_NAME)
     .add_buffer<float>(CSTM_TAG_VERT)
     .add_buffer<uint32_t>(CSTM_TAG_FACE)
     .add_buffer<uint32_t>(CSTM_TAG_IDX_MAP);

    

    if (pid == 0) {
        std::vector<fs::path> files;
        for (const auto& file : fs::directory_iterator(INPUT)) {
            if (fs::is_regular_file(file.status()))
                files.push_back(file);
        }

        auto file = "";
        qems::MeshMetaData metadata;
        std::vector<float> vertices;
        std::vector<uint32_t> faces;
        const auto& min = metadata.min_coords;
        const auto& max = metadata.max_coords;

        qems::import_mesh(file, metadata, vertices, faces);

        auto uniform_grid = mpi::UniformGridRow(
            vertices, faces, metadata.min_coords, 
            metadata.max_coords, START_PARTITIONS 
        );

        mpi::AsyncSend async_sender(layout);

        uint32_t current_idx = 0;
        uint32_t worker = 1;
        for (auto& cell : uniform_grid) {
            async_sender.wait();
            auto& msg = async_sender.get_message();

            //                                           old_index, new_index
            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID) = {current_idx, current_idx};
            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {START_PARTITIONS, START_PARTITIONS};

            float total_faces = static_cast<float>(faces.size()/3);
            uint32_t final_target = static_cast<uint32_t>(std::floor(total_faces * TARGET_FACES));
            msg.get_buffer<uint32_t>(CSTM_TAG_FINAL_TARGET) = { final_target };

            auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
            bb[0] = min.x(); bb[1] = min.y(); bb[2] = min.z();
            bb[3] = max.x(); bb[4] = max.y(); bb[5] = max.z();

            msg.get_buffer<char>(CSTM_TAG_NAME).clear();
            msg.get_buffer<char>(CSTM_TAG_NAME).assign(metadata.name.begin(), metadata.name.end());

            std::swap(msg.get_buffer<float>(CSTM_TAG_VERT), cell.vertices);
            std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_FACE), cell.faces);
            std::swap(msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP), cell.indices_mapping);
            async_sender.isend(worker); 
            current_idx++;
            worker++;
            if (worker >= num_procs) worker = 1;
        }

        //for (int i = 1; i < num_procs; i++)
            //mpi::sync_send(i, {}); 
        //
        //
        MPI_Barrier(MPI_COMM_WORLD);

    } else {

        mpi::AsyncSend async_sender(layout, 50);

        mpi::PackedMessage msg(layout);
        auto& id = msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID);
        auto& part_lvl = msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL);
        auto& final_target = msg.get_buffer<uint32_t>(CSTM_TAG_FINAL_TARGET);
        auto& bb = msg.get_buffer<double>(CSTM_TAG_BB);
        auto& name = msg.get_buffer<char>(CSTM_TAG_NAME);
        auto& vertices = msg.get_buffer<float>(CSTM_TAG_VERT);
        auto& faces = msg.get_buffer<uint32_t>(CSTM_TAG_FACE);
        auto& idx_mapping = msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP);

        qems::QEMMesh mesh;
        Eigen::Vector3d min = Eigen::Vector3d::Zero();
        Eigen::Vector3d max = Eigen::Vector3d::Zero();

        mesh.request_vertex_status();
        mesh.request_edge_status();
        mesh.request_face_status();
        mesh.request_halfedge_status();
        mesh.request_vertex_normals();
        mesh.request_face_normals();

        std::map<
            std::string,                                // The mesh computed
            std::map<                                   
                uint32_t,                               // Partitions level
                std::map<
                    uint32_t,                           // Cell ID 
                    std::vector<mpi::PackedMessage>     // The list of message to merge, when this list is full merge it
                >
            >
        > reduction_mapping;

        auto reduction_step = [&](uint32_t old_partitions, uint32_t new_partitions, std::string name) -> bool 
        {
            const uint32_t old_cell_id = msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID)[0];
            const uint32_t new_cell_id = msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID)[1];

            auto& partitions_map = reduction_mapping[name];
            auto& cell_map = partitions_map[new_partitions]; 
            auto& vector = cell_map[new_cell_id];

            vector.push_back(msg);

            const Eigen::Vector3i coords = mpi::UniformGridRow::get_cell_indices(new_cell_id, new_partitions);
            double scale = static_cast<double>(old_partitions) / static_cast<double>(new_partitions);
            
            int min_x = static_cast<int>(std::floor(coords.x() * scale));
            int min_y = static_cast<int>(std::floor(coords.y() * scale));
            int min_z = static_cast<int>(std::floor(coords.z() * scale));
            
            int max_x = static_cast<int>(std::floor((coords.x() + 1) * scale));
            int max_y = static_cast<int>(std::floor((coords.y() + 1) * scale));
            int max_z = static_cast<int>(std::floor((coords.z() + 1) * scale));
            
            int limit = static_cast<int>(old_partitions) - 1;
            
            min_x = std::max(0, std::min(limit, min_x));
            min_y = std::max(0, std::min(limit, min_y));
            min_z = std::max(0, std::min(limit, min_z));
            
            max_x = std::max(0, std::min(limit, max_x));
            max_y = std::max(0, std::min(limit, max_y));
            max_z = std::max(0, std::min(limit, max_z));
            
            uint32_t expected = static_cast<uint32_t>(std::max(0, max_x - min_x + 1)) * 
                                static_cast<uint32_t>(std::max(0, max_y - min_y + 1)) * 
                                static_cast<uint32_t>(std::max(0, max_z - min_z + 1));

            if (vector.size() == expected) {
                auto uniform_grid = mpi::UniformGridRow({}, {}, {}, min, max, old_partitions);
                for (auto& el : vector) {
                    uint32_t index = el.get_buffer<uint32_t>(CSTM_TAG_CELL_ID)[0];
                    auto& cell = uniform_grid.cells()[index];
                    cell.vertices = std::move(el.get_buffer<float>(CSTM_TAG_VERT));
                    cell.faces = std::move(el.get_buffer<uint32_t>(CSTM_TAG_FACE));
                    cell.indices_mapping = std::move(el.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP));
                }
                id[0] = new_cell_id;
                uniform_grid.merge_cells(vertices, faces, idx_mapping);
                LOG_INFO("PID {}, new_partitions: {}, cell_id: {} vertices: {} faces: {}", 
                         pid, new_partitions, new_cell_id, vertices.size()/3, faces.size()/3);
                return true;
            }
            return false;
        };

        std::queue<mpi::PackedMessage> tasks;
        while(true) {
            uint32_t old_partitions;
            uint32_t new_partitions;
            std::string str_name;

            if (tasks.empty()) {
                if(mpi::sync_recv(msg, MPI_ANY_SOURCE) == -1)
                    break;

                min.x() = bb[0]; min.y() = bb[1]; min.z() = bb[2]; 
                max.x() = bb[3]; max.y() = bb[4]; max.z() = bb[5]; 
                str_name = std::string(name.data(), name.size());
                old_partitions = part_lvl[0];
                new_partitions = part_lvl[1];

                if (old_partitions != new_partitions) {
                    if (!reduction_step(old_partitions, new_partitions, str_name))
                        continue;
                }
            } else {
                auto task = tasks.back();
                tasks.pop();

                vertices = std::move(task.get_buffer<float>(CSTM_TAG_VERT));
                faces = std::move(task.get_buffer<uint32_t>(CSTM_TAG_FACE));
                idx_mapping = std::move(task.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP));
                part_lvl = std::move(task.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL));
                id = std::move(task.get_buffer<uint32_t>(CSTM_TAG_CELL_ID));

                str_name = std::string(name.data(), name.size());
                old_partitions = part_lvl[0];
                new_partitions = part_lvl[1];
            }

            qems::row_data_to_mesh(vertices, faces, idx_mapping, mesh);
            {
                mesh.update_normals();
                std::vector<qems::QEMMesh::EdgeHandle> edges;

                for (auto vh : mesh.vertices()) {
                    auto coords = mesh.point(vh);
                    mesh.data(vh).Quadric = qems::compute_vertex_quadratic(mesh, vh);
                    mesh.data(vh).NodeIdx = mpi::UniformGridRow::get_vertex_index({coords[0], coords[1], coords[2]}, 
                                                                                  min, max, new_partitions);
                }

                for (auto eh : mesh.edges()) {
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    uint32_t idx0 = mesh.data(vh0).NodeIdx;
                    uint32_t idx1 = mesh.data(vh1).NodeIdx;

                    if (idx0 != idx1) {
                        mesh.data(vh0).Collasable = false;
                        mesh.data(vh1).Collasable = false;
                    } 
                } 

                for (auto eh : mesh.edges()) {
                    auto heh = mesh.halfedge_handle(eh, 0);
                    auto vh0 = mesh.from_vertex_handle(heh);
                    auto vh1 = mesh.to_vertex_handle(heh);

                    if (mesh.data(vh0).Collasable && mesh.data(vh1).Collasable) {
                        Eigen::Matrix4d Q = mesh.data(vh0).Quadric + mesh.data(vh1).Quadric;
                        Eigen::Vector4d newV = qems::compute_new_best_vertex(mesh, eh, Q);

                        mesh.data(eh).Error = newV.transpose() * Q * newV;
                        mesh.data(eh).NewVertex = newV;
                        edges.push_back(eh);
                    }
                }

                uint32_t collasable_faces = 0;
                for (auto fh : mesh.faces()) {
                    bool end = false;
                    for (auto fv_it = mesh.cfv_iter(fh); fv_it.is_valid(); ++fv_it) {
                        auto vh = *fv_it;
                        if (!mesh.data(vh).Collasable) {
                            end = true;
                            break;
                        }
                    }
                    if (end) continue;
                    collasable_faces++;
                }

                auto pq = qems::QEMPriorityQueue(qems::QEMEdgeCompare(mesh), edges);
                uint32_t local_target;
                if (new_partitions != 1) {
                    local_target = static_cast<uint32_t>(std::floor(static_cast<float>(collasable_faces) * 0.4));
                } else {
                    local_target = final_target[0];
                }

                qems::simplification(mesh, local_target, collasable_faces, pq);
                mesh.garbage_collection();
            }
            qems::mesh_to_row_data(mesh, vertices, faces, idx_mapping);

            uint32_t old_index = id[0];
            old_partitions = new_partitions;
            new_partitions = next_step(new_partitions);
            if (new_partitions == 0) {
                qems::export_mesh("out/final_test.ply", vertices, faces);
                exit(0);
            }

            auto uniform_grid = mpi::UniformGridRow(vertices, faces, idx_mapping, min, max, new_partitions);

            const auto coords = mpi::UniformGridRow::get_cell_indices(old_index, old_partitions);
            const float range = static_cast<float>(new_partitions) / static_cast<float>(old_partitions);
            
            uint32_t start_x = static_cast<uint32_t>(floorf(coords.x() * range));
            uint32_t start_y = static_cast<uint32_t>(floorf(coords.y() * range));
            uint32_t start_z = static_cast<uint32_t>(floorf(coords.z() * range));

            uint32_t end_x = static_cast<uint32_t>(floorf((coords.x() + 1) * range));
            uint32_t end_y = static_cast<uint32_t>(floorf((coords.y() + 1) * range));
            uint32_t end_z = static_cast<uint32_t>(floorf((coords.z() + 1) * range));

            if (end_x >= new_partitions) end_x = new_partitions - 1;
            if (end_y >= new_partitions) end_y = new_partitions - 1;
            if (end_z >= new_partitions) end_z = new_partitions - 1; 


            for (int x = start_x; x <= end_x; x++) {
                for (int y = start_y; y <= end_y; y++) {
                    for (int z = start_z; z <= end_z; z++) {
                        uint32_t new_index = uniform_grid.get_cell_index(x, y, z); 
                        uint32_t dest = (new_index % (num_procs - 1) + 1);
                        auto& cell = uniform_grid.cells()[new_index];

                        LOG_INFO("PID: {} receive: {} {} part: {}, send: {} ({}, {}, {}) part: {} dest: with verts: {}, faces: {}, mapping: {}", 
                                 pid, old_index, coords, old_partitions, new_index, x, y, z, new_partitions, dest,
                                 cell.vertices.size()/3,
                                 cell.faces.size()/3,
                                 cell.indices_mapping.size());

                        if (dest != pid) {
                            async_sender.wait(pid);

                            auto& send_msg = async_sender.get_message();
                            send_msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID) = {old_index, new_index};
                            send_msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {old_partitions, new_partitions};
                            send_msg.get_buffer<double>(CSTM_TAG_BB) = bb;
                            send_msg.get_buffer<char>(CSTM_TAG_NAME) = name;
                            send_msg.get_buffer<uint32_t>(CSTM_TAG_FINAL_TARGET) = final_target;

                            std::swap(send_msg.get_buffer<float>(CSTM_TAG_VERT), cell.vertices);
                            std::swap(send_msg.get_buffer<uint32_t>(CSTM_TAG_FACE), cell.faces);
                            std::swap(send_msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP), cell.indices_mapping);

                            async_sender.isend(dest);
                        } else {
                            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_ID) = {old_index, new_index};
                            msg.get_buffer<uint32_t>(CSTM_TAG_CELL_PART_LVL) = {old_partitions, new_partitions};
                            msg.get_buffer<float>(CSTM_TAG_VERT) = std::move(cell.vertices);
                            msg.get_buffer<uint32_t>(CSTM_TAG_FACE) = std::move(cell.faces);
                            msg.get_buffer<uint32_t>(CSTM_TAG_IDX_MAP) = std::move(cell.indices_mapping);

                            if(reduction_step(old_partitions, new_partitions, str_name))
                                tasks.push(msg);
                        }
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
    }


    MPI_Finalize();
    return 0;
}

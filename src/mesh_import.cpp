#include "logging.hpp"
#include "qem_mesh.hpp"
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <utils.hpp>
#include <mesh_import.hpp>
#include <vector>

namespace qems {

namespace detail {

template <ImportType type>
void import_ply(const std::filesystem::path path, MeshData& data) {
    std::ifstream file(path, std::ios::binary); 
    massert(file.is_open(), "Error: Failed to open PLY file.");

    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_vertices = 0;
    size_t num_faces = 0;
    bool header_end = false;
    std::streampos data_start_offset = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::stringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "format") {
            std::string format;
            ss >> format;
            massert(format == "ascii", 
                    "Binary PLY is not supported.");
        } else if (token == "element") {
            ss >> token;
            if (token == "vertex") {
                ss >> num_vertices;
            } else if (token == "face") {
                ss >> num_faces;
            }
        } else if (token == "end_header") {
            header_end = true;
            data_start_offset = file.tellg();
            break;
        }
    }

    massert(header_end, "Error: Malformed PLY header or missing end_header."); 
    massert(num_vertices > 0, "Error: PLY file does not declare vertices.");
    file.close();

    data.type = ".ply";
    data.name = path.filename();

    int max_threads = omp_get_max_threads();

    std::vector<size_t> lines_per_thread(max_threads, 0);
    std::vector<std::streampos> local_start_offsets(max_threads);
    std::vector<std::vector<float>> local_vertices(max_threads);
    std::vector<std::vector<uint32_t>> local_faces(max_threads);

    std::vector<Eigen::Vector3d> local_min(max_threads, 
        Eigen::Vector3d::Constant(std::numeric_limits<double>::max()));

    std::vector<Eigen::Vector3d> local_max(max_threads, 
        Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest()));

    uint64_t start_bytes = (uint64_t)data_start_offset;
    uint64_t total_bytes = (uint64_t)file_size;
    uint64_t data_bytes = total_bytes - start_bytes;

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();

        uint64_t chunk_size = data_bytes / n_threads;
        uint64_t my_start = start_bytes + (tid * chunk_size);
        uint64_t my_end = (tid == n_threads - 1) ? total_bytes : (my_start + chunk_size);

        std::ifstream t_file(path, std::ios::binary);
        t_file.seekg(my_start);

        if (tid > 0) {
            std::string dummy;
            std::getline(t_file, dummy);
        }
        
        std::streampos current_pos_stream = t_file.tellg();
        local_start_offsets[tid] = current_pos_stream;
     
        uint64_t current_byte_pos = (uint64_t)current_pos_stream;

        size_t my_lines = 0;
        std::string line_buf; 
        
        while (current_byte_pos < my_end) {
            if (!std::getline(t_file, line_buf)) break;
            current_byte_pos += line_buf.size() + 1; 
            my_lines++;
        }
        
        lines_per_thread[tid] = my_lines;
    }

    std::vector<size_t> global_line_start(max_threads, 0);
    size_t current_line_count = 0;
    for (int i = 0; i < max_threads; ++i) {
        global_line_start[i] = current_line_count;
        current_line_count += lines_per_thread[i];
    }

    auto parse_uint = [](char*& str) -> uint32_t {
        return std::strtoul(str, &str, 10);
    };
    
    auto parse_float = [](char*& str) -> float {
        return std::strtof(str, &str);
    };


    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t current_global_idx = global_line_start[tid];
        
        std::ifstream t_file(path, std::ios::binary);
        t_file.seekg(local_start_offsets[tid]);

        auto& vertices = local_vertices[tid];
        auto& faces = local_faces[tid];
        auto& min = local_min[tid];
        auto& max = local_max[tid];

        vertices.reserve((lines_per_thread[tid] > 0 ? lines_per_thread[tid] : 100) * 3);
        faces.reserve((lines_per_thread[tid] / 2) * 3);
       
        std::string line;
        for (size_t i = 0; i < lines_per_thread[tid]; ++i) {
            if(!std::getline(t_file, line)) break;
            
            char* line_ptr = line.data();
            if (current_global_idx < num_vertices) {
                float x = parse_float(line_ptr);
                float y = parse_float(line_ptr);
                float z = parse_float(line_ptr);

                if (x < min.x()) min.x() = x;
                if (y < min.y()) min.y() = y;
                if (z < min.z()) min.z() = z;
                if (x > max.x()) max.x() = x;
                if (y > max.y()) max.y() = y;
                if (z > max.z()) max.z() = z;

                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);

            } else {
                uint32_t count = parse_uint(line_ptr);
                
                if (count >= 3) {
                    uint32_t idx0 = parse_uint(line_ptr);
                    uint32_t idx_prev = parse_uint(line_ptr);

                    for (uint32_t k = 2; k < count; ++k) {
                        uint32_t idx_curr = parse_uint(line_ptr);
                        
                        faces.push_back(idx0);
                        faces.push_back(idx_prev);
                        faces.push_back(idx_curr);
                        
                        idx_prev = idx_curr;
                    }
                }
            }
            
            current_global_idx++;
        }
    }

    size_t vert_offset = 0;
    std::vector<float> &vertices = data.row_vertices;
    std::vector<uint32_t> &faces = data.row_faces;

    vertices.reserve(num_vertices * 3);
    for(int i=0; i<max_threads; ++i) {
        if (local_vertices[i].empty()) continue;

        vertices.insert(vertices.end(), 
                        local_vertices[i].begin(),
                        local_vertices[i].end());
        vert_offset += local_vertices[i].size();

        data.min_coords = data.min_coords.cwiseMin(local_min[i]);
        data.max_coords = data.max_coords.cwiseMax(local_max[i]);
    }

    size_t total_indices = 0;
    for(const auto& face : local_faces) 
        total_indices += face.size();
   
    faces.reserve(total_indices);
    for(int i=0; i<max_threads; ++i) {
        faces.insert(faces.end(), 
                     local_faces[i].begin(), 
                     local_faces[i].end());
    }


    if constexpr (type == ImportType::MESH_DATA ||
                  type == ImportType::ROW_MESH_DATA)
    {
        data.mesh.clear();
        data.mesh.reserve(num_vertices, num_vertices * 2, num_faces);

        for (size_t i = 0; i < data.row_vertices.size(); i+=3) {
            float x = data.row_vertices[i];
            float y = data.row_vertices[i+1];
            float z = data.row_vertices[i+2];
            data.mesh.add_vertex(QEMMesh::Point(x,y,z));
        }

        std::vector<QEMMesh::VertexHandle> face_vhs;
        face_vhs.reserve(3);

        for (size_t i = 0; i < data.row_faces.size(); i += 3) {
            face_vhs.clear();
            face_vhs.emplace_back(data.row_faces[i]);
            face_vhs.emplace_back(data.row_faces[i+1]);
            face_vhs.emplace_back(data.row_faces[i+2]);
            data.mesh.add_face(face_vhs);
        }

        if constexpr (type == ImportType::MESH_DATA) {
            data.row_vertices.clear();
            data.row_vertices.shrink_to_fit();
            data.row_faces.clear();
            data.row_faces.shrink_to_fit();
        }

    }
}

template <ImportType type>
void import_obj(const std::filesystem::path path, MeshData& data) {
    std::ifstream file(path);
    massert(file.is_open(), "Error: Failed to open OBJ file.");


    std::string line;
    size_t num_vertices = 0;
    size_t num_faces = 0;

    bool header_scanned = false;
    while (std::getline(file, line)) {
        if (line.rfind("# Vertices:", 0) == 0) {
            std::stringstream ss(line.substr(11));
            ss >> num_vertices;
        } else if (line.rfind("# Faces:", 0) == 0) {
            std::stringstream ss(line.substr(8));
            ss >> num_faces;
        } else if (line == "####") {
            if (num_vertices > 0) { 
                header_scanned = true;
                break;
            }
        }
    }

    massert(header_scanned, "Error: OBJ missing Meshlab header.");
    massert(num_vertices > 0, "Error: OBJ header has no vertex count.");

    data.type = ".obj";
    data.name = path.filename();

    if constexpr (type == ImportType::ROW_DATA ||
                  type == ImportType::ROW_MESH_DATA)
    {
        data.row_vertices.clear();
        data.row_faces.clear();
        data.row_vertices.reserve(num_vertices * 3);
        data.row_faces.reserve(num_faces * 3);
    }

    if constexpr (type == ImportType::MESH_DATA ||
                  type == ImportType::ROW_MESH_DATA)
    {
        data.mesh.clear();
        data.mesh.reserve(num_vertices, num_vertices * 2, num_faces);
    }


    data.max_coords = Eigen::Vector3d(std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max());
    
    data.max_coords = Eigen::Vector3d(std::numeric_limits<double>::lowest(),
                                      std::numeric_limits<double>::lowest(),
                                      std::numeric_limits<double>::lowest());

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            massert(!ss.fail(), "OBJ vertex parse error.");

            if (x < data.min_coords.x()) data.min_coords.x() = x;
            if (y < data.min_coords.y()) data.min_coords.y() = y;
            if (z < data.min_coords.z()) data.min_coords.z() = z;
    
            if (x > data.max_coords.x()) data.max_coords.x() = x;
            if (y > data.max_coords.y()) data.max_coords.y() = y;
            if (z > data.max_coords.z()) data.max_coords.z() = z;

            if constexpr (type == ImportType::MESH_DATA ||
                          type == ImportType::ROW_MESH_DATA)
            {
                data.mesh.add_vertex(QEMMesh::Point(x, y, z));
            }

            if constexpr (type == ImportType::ROW_DATA ||
                          type == ImportType::ROW_MESH_DATA)
            {
                data.row_vertices.push_back(x);
                data.row_vertices.push_back(y);
                data.row_vertices.push_back(z);
            }
        }

        else if (token == "f") {
            std::vector<uint32_t> idx;
            std::string v;
            while (ss >> v) {
                size_t slash = v.find('/');
                if (slash != std::string::npos)
                    v = v.substr(0, slash);

                int id = std::stoi(v) - 1;
                idx.push_back(id);
            }

            massert(idx.size() >= 3, "OBJ face < 3 vertices.");

            if constexpr (type == ImportType::MESH_DATA ||
                          type == ImportType::ROW_MESH_DATA)
            {
                std::vector<QEMMesh::VertexHandle> vhs;
                vhs.reserve(idx.size());
                for (uint32_t i : idx) vhs.emplace_back(i);
                data.mesh.add_face(vhs);
            }

            if constexpr (type == ImportType::ROW_DATA ||
                          type == ImportType::ROW_MESH_DATA)
            {
                for (size_t i = 1; i < idx.size() - 1; ++i) {
                    data.row_faces.push_back(idx[0]);
                    data.row_faces.push_back(idx[i]);
                    data.row_faces.push_back(idx[i + 1]);
                }
            }
        }
    }
}

template void import_obj<ImportType::MESH_DATA>(const std::filesystem::path path, MeshData& data);
template void import_obj<ImportType::ROW_DATA>(const std::filesystem::path path, MeshData& data);
template void import_obj<ImportType::ROW_MESH_DATA>(const std::filesystem::path path, MeshData& data);


template void import_ply<ImportType::ROW_DATA>(const std::filesystem::path path, MeshData& data);
template void import_ply<ImportType::MESH_DATA>(const std::filesystem::path path, MeshData& data);
template void import_ply<ImportType::ROW_MESH_DATA>(const std::filesystem::path path, MeshData& data);

}

template <ImportType type>
void import_mesh(const std::filesystem::path path, MeshData& data) {
    std::string ext = path.extension().string();

    if (ext == ".obj")
        detail::import_obj<type>(path, data);
    else if (ext == ".ply")
        detail::import_ply<type>(path, data);
    else
        massert(false, "Only .ply and .obj supported");
}

template void import_mesh<ImportType::ROW_DATA>(const std::filesystem::path path, MeshData& data);
template void import_mesh<ImportType::MESH_DATA>(const std::filesystem::path path, MeshData& data);
template void import_mesh<ImportType::ROW_MESH_DATA>(const std::filesystem::path path, MeshData& data);


//template <ImportType type>
//void import_ply(const std::filesystem::path path, MeshData& data) {
    //std::ifstream file(path);
    //massert(file.is_open(), "Error: Failed to open PLY file.");

    //size_t num_vertices = 0;
    //size_t num_faces = 0;
    //bool header_end = false;
    //std::streampos data_start_offset = 0;

    //std::string line;
    //while (std::getline(file, line)) {
        //std::stringstream ss(line);
        //std::string token;
        //ss >> token;

        //if (token == "format") {
            //std::string format;
            //ss >> format;
            //massert(format == "ascii", 
                    //"Binary PLY is not supported.");
        //} else if (token == "element") {
            //ss >> token;
            //if (token == "vertex") {
                //ss >> num_vertices;
            //} else if (token == "face") {
                //ss >> num_faces;
            //}
        //} else if (token == "end_header") {
            //header_end = true;
            //data_start_offset = file.tellg();
            //break;
        //}
    //}

    //massert(header_end, "Error: Malformed PLY header or missing end_header."); 
    //massert(num_vertices > 0, "Error: PLY file does not declare vertices.");
    //file.close();

    //data.type = ".ply";
    //data.name = path.filename();

    //if constexpr (type == ImportType::ROW_DATA || 
                  //type == ImportType::ROW_MESH_DATA) 
    //{
        //data.row_vertices.clear();
        //data.row_faces.clear();

        //data.row_vertices.reserve(num_vertices * 3);
        //data.row_faces.reserve(num_faces * 3);
    //} 

    //if constexpr (type == ImportType::MESH_DATA || 
                  //type == ImportType::ROW_MESH_DATA) 
    //{
        //data.mesh.clear();
        //data.mesh.reserve(num_vertices, num_vertices*2, num_faces);
    //}

    //data.max_coords = Eigen::Vector3d(std::numeric_limits<double>::max(),
                                      //std::numeric_limits<double>::max(),
                                      //std::numeric_limits<double>::max());
    
    //data.max_coords = Eigen::Vector3d(std::numeric_limits<double>::lowest(),
                                      //std::numeric_limits<double>::lowest(),
                                      //std::numeric_limits<double>::lowest());

    //std::vector<float> vertices;
    //vertices.reserve(num_vertices);

    //for (size_t i = 0; i < num_vertices; ++i) {
        //massert(!std::getline(file, line).fail(),
                //"Error: Unexpected EOF while reading vertices.");
        
        //std::stringstream ss(line);
        //float x, y, z;
        //ss >> x >> y >> z;

        //massert(!ss.fail(), "Error: Vertex parsing failed (non-numeric data?).");

        //if (x < data.min_coords.x()) data.min_coords.x() = x;
        //if (y < data.min_coords.y()) data.min_coords.y() = y;
        //if (z < data.min_coords.z()) data.min_coords.z() = z;
    
        //if (x > data.max_coords.x()) data.max_coords.x() = x;
        //if (y > data.max_coords.y()) data.max_coords.y() = y;
        //if (z > data.max_coords.z()) data.max_coords.z() = z;

        //vertices.push_back(x);
        //vertices.push_back(y);
        //vertices.push_back(z);
    //}

    //for (size_t i = 0; i < num_faces; ++i) {
        //massert(!std::getline(file, line).fail(),
                //"Error: Unexpected EOF while reading faces.");
        
        //std::stringstream ss(line);
        //int count;
        //ss >> count;
        //massert(count >= 3, "Error: Degenerate face < 3 vertices.");

        //std::vector<uint32_t> face_indices(count);
        //for (int k = 0; k < count; ++k) {
            //ss >> face_indices[k];
        //}
        //massert(!ss.fail(), "Error: Face index parsing failed.");

        //if constexpr (type == ImportType::MESH_DATA || 
                      //type == ImportType::ROW_MESH_DATA) 
        //{
            //std::vector<QEMMesh::VertexHandle> face_vhs;
            //face_vhs.reserve(count);

            //for (uint32_t idx : face_indices)
                //face_vhs.emplace_back(idx); 
            
            //data.mesh.add_face(face_vhs);
        //}

        //if constexpr (type == ImportType::ROW_DATA || 
                      //type == ImportType::ROW_MESH_DATA) 
        //{
            //for (int k = 1; k < count - 1; ++k) {
                //data.row_faces.push_back(face_indices[0]);
                //data.row_faces.push_back(face_indices[k]);
                //data.row_faces.push_back(face_indices[k + 1]);
            //}
        //}
    //}
//}

}

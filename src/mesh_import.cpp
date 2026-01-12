#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <utils.hpp>
#include <mesh_import.hpp>
#include <vector>

namespace qems {

namespace detail {

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
}

void import_obj(const std::filesystem::path path, MeshData& data) 
{
    massert(false, ".obj not supported");
}

}

void import_mesh(const std::filesystem::path path, MeshData& data) {
    std::string ext = path.extension().string();

    if (ext == ".obj")
        detail::import_obj(path, data);
    else if (ext == ".ply")
        detail::import_ply(path, data);
    else
        massert(false, "Only .ply and .obj supported");
}

}

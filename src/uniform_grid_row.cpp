#include <uniform_grid_row.hpp>

namespace mpi {

void UniformGridRow::merge_cells(const std::vector<Cell>& cells,
                                 std::vector<float>& vertices,
                                 std::vector<uint32_t>& faces,
                                 std::vector<uint32_t>& verts_mapping) 
{
    struct Array3Hash {
        std::size_t operator()(const std::array<uint32_t, 3>& k) const {
            return std::hash<uint32_t>()(k[0]) ^ 
                   (std::hash<uint32_t>()(k[1]) << 1) ^ 
                   (std::hash<uint32_t>()(k[2]) << 2);
        }
    };

    vertices.clear();    
    faces.clear();
    verts_mapping.clear();

    std::unordered_map<uint32_t, uint32_t> mapping;
    std::unordered_set<std::array<uint32_t, 3>, Array3Hash> unique_faces;
    for (uint32_t c = 0; c < cells.size(); ++c) {
        const auto& cell = cells[c];
        if (cell.faces.empty()) 
            continue;

        std::vector<uint32_t> local_to_merged(cell.vertices.size() / 3);

        for (size_t i = 0; i < cell.indices_mapping.size(); ++i) {
            uint32_t original_id = cell.indices_mapping[i];

            auto it = mapping.find(original_id);
            if (it == mapping.end()) {
                uint32_t new_idx = static_cast<uint32_t>(vertices.size() / 3);
                vertices.push_back(cell.vertices[i * 3]);
                vertices.push_back(cell.vertices[i * 3 + 1]);
                vertices.push_back(cell.vertices[i * 3 + 2]);

                verts_mapping.push_back(original_id);
                mapping[original_id] = new_idx;
                local_to_merged[i] = new_idx;
            } else {
                local_to_merged[i] = it->second;
            }
        }

        for (size_t i = 0; i < cell.faces.size(); i += 3) {
            uint32_t local_idx0 = cell.faces[i];
            uint32_t local_idx1 = cell.faces[i + 1];
            uint32_t local_idx2 = cell.faces[i + 2];

            uint32_t global_idx0 = local_to_merged[local_idx0];
            uint32_t global_idx1 = local_to_merged[local_idx1];
            uint32_t global_idx2 = local_to_merged[local_idx2];

            std::array<uint32_t, 3> face_key = {global_idx0, global_idx1, global_idx2};
            std::sort(face_key.begin(), face_key.end());

            if (unique_faces.find(face_key) == unique_faces.end()) {
                unique_faces.insert(face_key);

                faces.push_back(global_idx0);
                faces.push_back(global_idx1);
                faces.push_back(global_idx2);
            }
        }
    }
} 

void UniformGridRow::init(const std::vector<float>& vertices, 
                          const std::vector<uint32_t>& faces,
                          const std::vector<uint32_t>& mapping)
{
    std::vector<std::unordered_map<uint32_t, uint32_t>> cells_lookup(cells_.size());
    for(std::size_t i = 0; i < faces.size(); i += 3) {
        uint32_t idx0 = faces[i];
        uint32_t idx1 = faces[i + 1];
        uint32_t idx2 = faces[i + 2];

        Eigen::Vector3d p0(vertices[idx0*3], vertices[idx0*3+1], vertices[idx0*3+2]);
        Eigen::Vector3d p1(vertices[idx1*3], vertices[idx1*3+1], vertices[idx1*3+2]);
        Eigen::Vector3d p2(vertices[idx2*3], vertices[idx2*3+1], vertices[idx2*3+2]);

        uint32_t c0 = get_vertex_index(p0);
        uint32_t c1 = get_vertex_index(p1);
        uint32_t c2 = get_vertex_index(p2);

        std::array<uint32_t, 3> involved_cells = {c0, c1, c2};
        std::sort(std::begin(involved_cells), std::end(involved_cells));
        auto last = std::unique(std::begin(involved_cells), std::end(involved_cells));
        std::size_t unique_count = std::distance(involved_cells.begin(), last);

        if (unique_count == 1)
            total_collasable_faces_++;

        for (auto it = std::begin(involved_cells); it != last; ++it) {
            uint32_t cell_idx = *it;

            auto& cell = cells_[cell_idx];
            auto& lookup = cells_lookup[cell_idx];

            auto get_or_add_vertex = [&](uint32_t input_idx, const Eigen::Vector3d& p) -> uint32_t {
                if (lookup.find(input_idx) != lookup.end()) {
                    return lookup[input_idx];
                }

                uint32_t local_idx = cell.vertices.size() / 3;
                cell.vertices.push_back(p.x());
                cell.vertices.push_back(p.y());
                cell.vertices.push_back(p.z());
                    
                uint32_t original_global_id = mapping[input_idx];
                cell.indices_mapping.push_back(original_global_id);

                lookup[input_idx] = local_idx;
                return local_idx;
            };

            uint32_t local0 = get_or_add_vertex(idx0, p0);    
            uint32_t local1 = get_or_add_vertex(idx1, p1);
            uint32_t local2 = get_or_add_vertex(idx2, p2);

            cell.faces.push_back(local0);
            cell.faces.push_back(local1);
            cell.faces.push_back(local2);
        }
    }
}

}
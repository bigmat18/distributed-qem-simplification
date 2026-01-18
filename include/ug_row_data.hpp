#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <set>

#include <Eigen/Dense>

namespace mpi {

class UniformGridRow {

    struct Cell {
        std::vector<uint32_t> indices_mapping;
        std::vector<float> vertices;
        std::vector<uint32_t> faces;
    };

    Eigen::Vector3d min_coords_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_coords_ = Eigen::Vector3d::Zero();

    uint32_t num_split_ = 4;
    std::vector<Cell> cells_;
public:

    UniformGridRow(const std::vector<float>& vertices, 
                   const std::vector<uint32_t>& faces,
                   Eigen::Vector3d min_coords, 
                   Eigen::Vector3d max_coords, 
                   uint32_t num_split = 4) : 
        min_coords_(min_coords), 
        max_coords_(max_coords), 
        num_split_(num_split),
        cells_(num_split_ * num_split_ * num_split_) 
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

            uint32_t involved_cells[3] = {c0, c1, c2};
            std::sort(std::begin(involved_cells), std::end(involved_cells));
            auto last = std::unique(std::begin(involved_cells), std::end(involved_cells));

            for (auto it = std::begin(involved_cells); it != last; ++it) {
                uint32_t cell_idx = *it;

                auto& cell = cells_[cell_idx];
                auto& lookup = cells_lookup[cell_idx];

                auto get_or_add_vertex = [&](uint32_t global_idx, const Eigen::Vector3d& p) -> uint32_t {
                    if (lookup.find(global_idx) != lookup.end()) {
                        return lookup[global_idx];
                    }

                    uint32_t local_idx = cell.vertices.size() / 3;
                    cell.vertices.push_back(p.x());
                    cell.vertices.push_back(p.y());
                    cell.vertices.push_back(p.z());
                    
                    cell.indices_mapping.push_back(global_idx);

                    lookup[global_idx] = local_idx;
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

    auto begin() noexcept { return cells_.begin(); }
    auto end()   noexcept { return cells_.end(); }

    auto begin()  const noexcept { return cells_.begin(); }
    auto end()    const noexcept { return cells_.end(); }
    auto cbegin() const noexcept { return cells_.cbegin(); }
    auto cend()   const noexcept { return cells_.cend(); }

    std::size_t size() const { return cells_.size(); }


    void merge_cells(std::vector<float>& vertices, 
                     std::vector<uint32_t>& faces)
    {
        vertices.clear();
        faces.clear();
    
        std::unordered_map<uint32_t, uint32_t> original_to_new_map;
        std::set<std::array<uint32_t, 3>> processed_faces;
    
        for (const auto& cell : cells_) {
            std::vector<uint32_t> local_to_merged(cell.vertices.size() / 3);
    
            for (size_t i = 0; i < cell.indices_mapping.size(); ++i) {
                uint32_t original_id = cell.indices_mapping[i];
                
                if (original_to_new_map.find(original_id) == original_to_new_map.end()) {
                    uint32_t new_idx = static_cast<uint32_t>(vertices.size() / 3);
                    vertices.push_back(cell.vertices[i*3]);
                    vertices.push_back(cell.vertices[i*3+1]);
                    vertices.push_back(cell.vertices[i*3+2]);
                    
                    original_to_new_map[original_id] = new_idx;
                }
                local_to_merged[i] = original_to_new_map[original_id];
            }
    
            for (size_t i = 0; i < cell.faces.size(); i += 3) {
                std::array<uint32_t, 3> face_key = {
                    cell.indices_mapping[cell.faces[i]],
                    cell.indices_mapping[cell.faces[i+1]],
                    cell.indices_mapping[cell.faces[i+2]]
                };
                
                std::sort(face_key.begin(), face_key.end());
    
                if (processed_faces.insert(face_key).second) {
                    faces.push_back(local_to_merged[cell.faces[i]]);
                    faces.push_back(local_to_merged[cell.faces[i+1]]);
                    faces.push_back(local_to_merged[cell.faces[i+2]]);
                }
            }
        }
    }

private:

    inline uint32_t get_vertex_index(const Eigen::Vector3d vertex) {
        Eigen::Vector3d block_size = (max_coords_ - min_coords_) / num_split_;
        uint32_t x = std::min(static_cast<uint32_t>(std::floor(vertex.x() / block_size.x())), num_split_ - 1);
        uint32_t y = std::min(static_cast<uint32_t>(std::floor(vertex.y() / block_size.y())), num_split_ - 1);
        uint32_t z = std::min(static_cast<uint32_t>(std::floor(vertex.z() / block_size.z())), num_split_ - 1);
        uint32_t index = x + (y * num_split_) + (z * num_split_ * num_split_);

        return index;
    }
};

}

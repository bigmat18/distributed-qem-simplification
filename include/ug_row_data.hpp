#pragma once

#include <cstddef>
#include <utils.hpp>
#include <cstdint>
#include <numeric>
#include <vector>
#include <unordered_set>

#include <Eigen/Dense>

namespace mpi {

class UniformGridRow {
    using BoundingBox = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

    struct Cell {
        std::vector<uint32_t> indices_mapping;
        std::vector<float> vertices;
        std::vector<uint32_t> faces;
    };

    Eigen::Vector3d min_coords_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_coords_ = Eigen::Vector3d::Zero();

    uint32_t num_split_ = 4;
    std::vector<Cell> cells_;
    uint32_t total_collasable_faces_ = 0;
public:
    UniformGridRow() = default;

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
        std::vector<uint32_t> mapping(vertices.size() / 3);   
        std::iota(mapping.begin(), mapping.end(), 0);
        init(vertices, faces, mapping);
    }

    UniformGridRow(const std::vector<float>& vertices, 
                   const std::vector<uint32_t>& faces,
                   const std::vector<uint32_t>& mapping,
                   Eigen::Vector3d min_coords, 
                   Eigen::Vector3d max_coords, 
                   uint32_t num_split = 4) : 
        min_coords_(min_coords), 
        max_coords_(max_coords), 
        num_split_(num_split),
        cells_(num_split_ * num_split_ * num_split_) 
    {
        init(vertices, faces, mapping);
    }

    auto begin() noexcept { return cells_.begin(); }
    auto end()   noexcept { return cells_.end(); }

    auto begin()  const noexcept { return cells_.begin(); }
    auto end()    const noexcept { return cells_.end(); }
    auto cbegin() const noexcept { return cells_.cbegin(); }
    auto cend()   const noexcept { return cells_.cend(); }

    std::size_t size() const { return cells_.size(); }

    auto& cells() { return cells_; }


    void merge_cells(std::vector<float>& vertices, 
                     std::vector<uint32_t>& faces,
                     std::vector<uint32_t>& verts_mapping)
    { 
        merge_cells(cells_, vertices, faces, verts_mapping); 
    }


    void merge_cells(std::vector<float>& vertices, 
                     std::vector<uint32_t>& faces)
    {
        std::vector<uint32_t> tmp_verts_mapping;
        merge_cells(cells_, vertices, faces, tmp_verts_mapping); 
    }

    void merge_cells(const std::vector<Cell>& cells,
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

    inline uint32_t get_vertex_index(const Eigen::Vector3d& vertex) {
        return UniformGridRow::get_vertex_index(vertex, min_coords_, max_coords_, num_split_);
    }

    inline static uint32_t get_vertex_index(const Eigen::Vector3d& vertex, 
                                            const Eigen::Vector3d& min, 
                                            const Eigen::Vector3d& max, 
                                            const uint32_t num_split) 
    {
        Eigen::Vector3d block_size = (max - min) / num_split;

        Eigen::Vector3d local_pos = vertex - min; 

        uint32_t x = std::min(static_cast<uint32_t>(std::floor(local_pos.x() / block_size.x())), num_split - 1);
        uint32_t y = std::min(static_cast<uint32_t>(std::floor(local_pos.y() / block_size.y())), num_split - 1);
        uint32_t z = std::min(static_cast<uint32_t>(std::floor(local_pos.z() / block_size.z())), num_split - 1);

        uint32_t index = x + (y * num_split) + (z * num_split * num_split);

        return index;
    }


    inline uint32_t get_cell_index(uint32_t x, uint32_t y, uint32_t z) const {
        massert(x < num_split_ && x >= 0, "X value is wrong");
        massert(y < num_split_ && y >= 0, "Y value is wrong");
        massert(z < num_split_ && z >= 0, "Z value is wrong");
        return x + (y * num_split_) + (z * num_split_ * num_split_);
    }

    inline Eigen::Vector3i get_cell_indices(uint32_t index) const {
        return UniformGridRow::get_cell_indices(index, num_split_);
    }

    inline static Eigen::Vector3i get_cell_indices(uint32_t index, 
                                                   uint32_t num_split)  
    {
        uint32_t slice_size = num_split * num_split;

        uint32_t z = index / slice_size;
        uint32_t remainder = index % slice_size;

        uint32_t y = remainder / num_split;
        uint32_t x = remainder % num_split;

        return Eigen::Vector3i(x, y, z);

    }

    inline uint32_t total_collasable_faces() const { return total_collasable_faces_; }

private:

    void init(const std::vector<float>& vertices, 
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
};

}

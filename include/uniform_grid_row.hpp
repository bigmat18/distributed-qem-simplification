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
                     std::vector<uint32_t>& verts_mapping); 

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
              const std::vector<uint32_t>& mapping);
};

}

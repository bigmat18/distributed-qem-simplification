#include <cstdint>
#include <utils.hpp>
#include <octree.hpp>

namespace qems {

uint32_t Octree::add_vertex(const QEMMesh& mesh, const QEMMesh::VertexHandle vh) {
    Eigen::Vector3d max = max_coords_;
    Eigen::Vector3d min = min_coords_;
    uint32_t offset = 0;
    

    while (true) {
        auto indices = get_vertex_indices(mesh, vh, max, min);
        auto idx = offset + indices.w();
        Node& node = tree_[idx];

        if (node.is_leaf) {
            node.vertices.push_back(vh);
            return idx;
        } else {
            std::size_t jump = 8;
            for (int i = 0; i < indices.w(); i++)
                if (!tree_[offset + i].is_leaf)
                    jump += 8;

            offset += jump;
            min = node.min_coords;
            max = node.max_coords;
        }
    }
    std::unreachable();
    massert(false, "Unreachable code in add_vertex");
    return -1;
}

void Octree::add_edge(const QEMMesh& mesh, const QEMMesh::EdgeHandle eh) {
    auto heh = mesh.halfedge_handle(eh);
    auto vh1 = mesh.from_vertex_handle(heh);
    auto vh2 = mesh.to_vertex_handle(heh);

    if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
        auto idx1 = mesh.data(vh1).NodeIdx;
        auto idx2 = mesh.data(vh2).NodeIdx;
        massert(idx1 == idx2,"Vertex in differents node both collasable are not allowed");
        tree_[mesh.data(vh1).NodeIdx].edges.push_back(eh);
    }
}

void Octree::increment_collasable_faces(const QEMMesh& mesh, const QEMMesh::FaceHandle fh) {
    for (const auto& vh : mesh.fv_range(fh)) {
        if (!mesh.data(vh).Collasable)
            return;
    }

    auto vh = *mesh.cfv_iter(fh);
    tree_[mesh.data(vh).NodeIdx].collasable_faces++;
    total_collasable_faces_++;
}

void Octree::merge(const Octree& other) {
    for (size_t i = 0; i < tree_.size(); i++) {
        if (!tree_[i].is_leaf)
            continue;

        massert(tree_[i].is_leaf == other.tree_[i].is_leaf,
                "Can not be merge different octree");

        tree_[i].vertices.insert(
            tree_[i].vertices.end(),
            other.tree_[i].vertices.begin(),
            other.tree_[i].vertices.end()
        );

        tree_[i].edges.insert(
            tree_[i].edges.end(),
            other.tree_[i].edges.begin(),
            other.tree_[i].edges.end()
        );

        tree_[i].collasable_faces += other.tree_[i].collasable_faces;
    }
    total_collasable_faces_ += other.total_collasable_faces_;
}

void Octree::normalize(QEMMesh& mesh) {
    #pragma omp parallel for schedule(static)
    for(auto& node : tree_)
        node.collasable_faces = 0;

    total_collasable_faces_ = 0;

    for (int i = 0; i < tree_.size(); i++) {
        if (!tree_[i].is_leaf)
            continue;

        if (tree_[i].vertices.size() > limit_)
            split(mesh, i);
    }

    #pragma omp parallel
    {
        uint32_t local_total_collasable_faces = 0;
        std::vector<uint32_t> local_faces_count(tree_.size(), 0);

        #pragma omp for schedule(static) nowait
        for(size_t i = 0; i < mesh.n_faces(); i++) {
            auto fh = QEMMesh::FaceHandle(i);

            bool collasable = true; 
            for (const auto& vh : mesh.fv_range(fh))
            collasable &= mesh.data(vh).Collasable;

            if (collasable) {
                auto vh = *mesh.cfv_iter(fh);
                local_faces_count[mesh.data(vh).NodeIdx]++;
                local_total_collasable_faces++;
            }
        }

        for (size_t i = 0; i < tree_.size(); i++) {
            if (local_faces_count[i] != 0) {
                #pragma omp atomic
                tree_[i].collasable_faces += local_faces_count[i];

                #pragma omp atomic
                total_collasable_faces_ += local_total_collasable_faces;
            }
        }
    }
}

void Octree::export_mesh(const std::filesystem::path path) {
    QEMMesh mesh;
    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.clear();

    for (const auto& node : tree_) {
        if (!node.is_leaf)
            continue;

        std::array<Eigen::Vector3d, 8> V;
        const float xmin = node.min_coords.x(), ymin = node.min_coords.y(), zmin = node.min_coords.z();
        const float xmax = node.max_coords.x(), ymax = node.max_coords.y(), zmax = node.max_coords.z();

        V[0] = {xmin, ymin, zmin};
        V[1] = {xmax, ymin, zmin};
        V[2] = {xmax, ymax, zmin};
        V[3] = {xmin, ymax, zmin};
        V[4] = {xmin, ymin, zmax};
        V[5] = {xmax, ymin, zmax};
        V[6] = {xmax, ymax, zmax};
        V[7] = {xmin, ymax, zmax};

        std::array<QEMMesh::VertexHandle, 8> vh;
        for (int i = 0; i < 8; ++i)
            vh[i] = mesh.add_vertex(QEMMesh::Point(V[i].x(), V[i].y(), V[i].z()));

        auto addTri = [&](int a, int b, int c) {
            std::vector<QEMMesh::VertexHandle> face_vhandles;
            face_vhandles.reserve(3);
            face_vhandles.push_back(vh[a]);
            face_vhandles.push_back(vh[b]);
            face_vhandles.push_back(vh[c]);
            mesh.add_face(face_vhandles);
        };

        addTri(0, 1, 2);
        addTri(0, 2, 3);

        addTri(4, 6, 5);
        addTri(4, 7, 6);

        addTri(0, 5, 1);
        addTri(0, 4, 5);

        addTri(3, 2, 6);
        addTri(3, 6, 7);

        addTri(0, 3, 7);
        addTri(0, 7, 4);

        addTri(1, 6, 2);
        addTri(1, 5, 6);

    }

    massert(OpenMesh::IO::write_mesh(mesh, path), "Error in octree export!");
}

void Octree::split(QEMMesh& mesh, uint32_t index) {
    std::vector<QEMMesh::VertexHandle> vertices = std::move(tree_[index].vertices);
    std::vector<QEMMesh::EdgeHandle> edges = std::move(tree_[index].edges);

    Eigen::Vector3d min = tree_[index].min_coords;
    Eigen::Vector3d max = tree_[index].max_coords;
    Eigen::Vector3d center = (min + max) * 0.5f;

    tree_[index].is_leaf = false;
    tree_[index].collasable_faces = 0; 

    size_t first_child_idx = tree_.size();
    tree_.insert(tree_.end(), 8, Node());

    for(size_t i = 0; i < 8; i++) {
        Node& child = tree_[first_child_idx + i];
        BoundingBox bb = compute_bounding_box(i, center, min, max);
        child.min_coords = bb.first;
        child.max_coords = bb.second;
    }

    #pragma omp parallel
    {
        std::vector<QEMMesh::VertexHandle> local_vertices[8];
        std::vector<QEMMesh::EdgeHandle> local_edges[8];

        #pragma omp for schedule(static) nowait
        for (const auto& vh : vertices) {
            auto indices = get_vertex_indices(mesh, vh, max, min);
            size_t child_local = indices.w(); 
            size_t child_global = first_child_idx + child_local;

            local_vertices[child_local].push_back(vh);
            mesh.data(vh).NodeIdx = child_global;
        } 

        for (int i = 0; i < 8; ++i) {
            if (!local_vertices[i].empty()) {
                std::size_t idx = first_child_idx + i;
                #pragma omp critical
                {
                    tree_[idx].vertices.insert(
                        tree_[idx].vertices.end(),
                        local_vertices[i].begin(),
                        local_vertices[i].end()
                    );
                }
            }
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (const auto& eh : edges) {    
            auto heh = mesh.halfedge_handle(eh, 0);
            auto vh0 = mesh.from_vertex_handle(heh);
            auto vh1 = mesh.to_vertex_handle(heh);

            size_t idx0 = mesh.data(vh0).NodeIdx;
            size_t idx1 = mesh.data(vh1).NodeIdx;

            if (idx0 != idx1) {
                mesh.data(vh0).Collasable = false;
                mesh.data(vh1).Collasable = false;
            }
        }

        #pragma omp for schedule(static) nowait
        for (const auto& eh : edges) {
            auto heh = mesh.halfedge_handle(eh);
            auto vh1 = mesh.from_vertex_handle(heh);
            auto vh2 = mesh.to_vertex_handle(heh);

            if (mesh.data(vh1).Collasable && mesh.data(vh2).Collasable) {
                auto idx1 = mesh.data(vh1).NodeIdx;
                auto idx2 = mesh.data(vh2).NodeIdx;
                massert(idx1 == idx2,"Vertex in differents node both collasable are not allowed");

                size_t local_idx = idx1 - first_child_idx; // 0-7
                local_edges[local_idx].push_back(eh);
            }
        }

        for (int i = 0; i < 8; ++i) {
            if (!local_edges[i].empty()) {
                std::size_t idx = first_child_idx + i;
                #pragma omp critical
                {

                    tree_[idx].edges.insert(
                        tree_[idx].edges.end(),
                        local_edges[i].begin(),
                        local_edges[i].end()
                    );
                }
            }
        }
    }
}

} // namespace qems

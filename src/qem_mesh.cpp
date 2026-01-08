#include "qem_mesh.hpp"
#include <cstddef>
#include <cstdint>
#include <qem_simp.hpp>

namespace qems {

void row_data_to_mesh(const std::vector<float>& vertices, 
                      const std::vector<uint32_t>& faces,
                      QEMMesh& mesh)
{
    const uint32_t num_vertices = vertices.size();
    const uint32_t num_faces = faces.size();

    mesh.clear();
    mesh.reserve(num_vertices, num_vertices * 2, num_faces);

    for (size_t i = 0; i < num_vertices; i+=3) {
        float x = vertices[i];
        float y = vertices[i+1];
        float z = vertices[i+2];
        mesh.add_vertex(QEMMesh::Point(x,y,z));
    }

    std::vector<QEMMesh::VertexHandle> face_vhs;
    face_vhs.reserve(3);

    for (size_t i = 0; i < num_faces; i += 3) {
        face_vhs.clear();
        face_vhs.emplace_back(faces[i]);
        face_vhs.emplace_back(faces[i+1]);
        face_vhs.emplace_back(faces[i+2]);
        mesh.add_face(face_vhs);
    }

}

void mesh_to_row_data(const QEMMesh& mesh,
                      std::vector<float> &vertices, 
                      std::vector<uint32_t> &faces)
{
    vertices.clear();
    faces.clear();

    vertices.reserve(mesh.n_vertices() * 3);
    faces.reserve(mesh.n_faces() * 3);

    for (std::size_t i = 0; i < mesh.n_vertices(); ++i) {
        const auto vh = QEMMesh::VertexHandle(i);
        if (mesh.status(vh).deleted())
            continue;

        const auto coords = mesh.point(vh);

        vertices.push_back(coords[0]);
        vertices.push_back(coords[1]);
        vertices.push_back(coords[2]);
    }

    for (std::size_t i = 0; i < mesh.n_faces(); ++i) {
        const auto fh = QEMMesh::FaceHandle(i);
        if (mesh.status(fh).deleted())
            continue;

        for (const auto fv : mesh.fv_range(fh)) {
            faces.push_back(static_cast<uint32_t>(fv.idx()));
        }
    }

}

}

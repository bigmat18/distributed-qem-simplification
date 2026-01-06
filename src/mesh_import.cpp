#include "logging.hpp"
#include "qem_mesh.hpp"
#include <fstream>
#include <utils.hpp>
#include <mesh_import.hpp>

namespace qems {

namespace detail {


template <ImportType type>
void import_ply(const std::filesystem::path path, MeshData& data) {
    std::ifstream file(path);
    massert(file.is_open(), "Error: Failed to open PLY file.");

    size_t num_vertices = 0;
    size_t num_faces = 0;
    bool header_end = false;

    std::string line;
    while (std::getline(file, line)) {
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
            break;
        }
    }

    massert(header_end, "Error: Malformed PLY header or missing end_header."); 
    massert(num_vertices > 0, "Error: PLY file does not declare vertices.");

    data.type = ".ply";
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
        data.mesh.reserve(num_vertices, num_vertices*2, num_faces);
    }


    for (size_t i = 0; i < num_vertices; ++i) {
        massert(!std::getline(file, line).fail(),
                "Error: Unexpected EOF while reading vertices.");
        
        std::stringstream ss(line);
        float x, y, z;
        ss >> x >> y >> z;

        massert(!ss.fail(), "Error: Vertex parsing failed (non-numeric data?).");
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

    for (size_t i = 0; i < num_faces; ++i) {
        massert(!std::getline(file, line).fail(),
                "Error: Unexpected EOF while reading faces.");
        
        std::stringstream ss(line);
        int count;
        ss >> count;
        massert(count >= 3, "Error: Degenerate face < 3 vertices.");

        std::vector<uint32_t> face_indices(count);
        for (int k = 0; k < count; ++k) {
            ss >> face_indices[k];
        }
        massert(!ss.fail(), "Error: Face index parsing failed.");

        if constexpr (type == ImportType::MESH_DATA || 
                      type == ImportType::ROW_MESH_DATA) 
        {
            std::vector<QEMMesh::VertexHandle> face_vhs;
            face_vhs.reserve(count);

            for (uint32_t idx : face_indices)
                face_vhs.emplace_back(idx); 
            
            data.mesh.add_face(face_vhs);
        }

        if constexpr (type == ImportType::ROW_DATA || 
                      type == ImportType::ROW_MESH_DATA) 
        {
            for (int k = 1; k < count - 1; ++k) {
                data.row_faces.push_back(face_indices[0]);
                data.row_faces.push_back(face_indices[k]);
                data.row_faces.push_back(face_indices[k + 1]);
            }
        }
    }
}

template <ImportType type>
void import_obj(const std::filesystem::path path, MeshData& data) {
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



//bool import_mesh_data(std::filesystem::path path, 
                      //std::vector<float>& vertices, 
                      //std::vector<uint32_t>& faces) 
//{
    //vertices.clear();
    //faces.clear();

    //std::ifstream file(path);
    //massert(file.is_open(), "Error: Failed to open PLY file.");

    //std::string line;
    //size_t vertexCount = 0;
    //size_t faceCount = 0;
    //bool headerEnded = false;

    //while (std::getline(file, line)) {
        //std::stringstream ss(line);
        //std::string token;
        //ss >> token;


        //if (token == "format") {
            //std::string formatType;
            //ss >> formatType;
            //massert(formatType == "ascii", 
                    //"Error: Input file is not ASCII. Binary PLY is not supported.");
        //} else if (token == "element") {
            //ss >> token;
            //if (token == "vertex") {
                //ss >> vertexCount;
            //} else if (token == "face") {
                //ss >> faceCount;
            //}
        //} else if (token == "end_header") {
            //headerEnded = true;
            //break;
        //}
    //}

    //massert(headerEnded, "Error: Malformed PLY header or missing end_header."); 
    //massert(vertexCount > 0, "Error: PLY file does not declare vertices.");

    //vertices.reserve(vertexCount * 3);
    //faces.reserve(faceCount * 3);

    //for (size_t i = 0; i < vertexCount; ++i) {
        //massert(!std::getline(file, line).fail(),
                //"Error: Unexpected EOF while reading vertices.");
        
        //std::stringstream ss(line);
        //float x, y, z;
        //ss >> x >> y >> z;

        //massert(!ss.fail(), "Error: Vertex parsing failed (non-numeric data?).");

        //vertices.push_back(x);
        //vertices.push_back(y);
        //vertices.push_back(z);
        //std::println("{} {} {}", x,y,z);
    //}

    //for (size_t i = 0; i < faceCount; ++i) {
        //massert(!std::getline(file, line).fail(),
                //"Error: Unexpected EOF while reading faces.");
        
        //std::stringstream ss(line);
        //int count;
        //ss >> count;

        //massert(count >= 3, "Error: Degenerate face found with < 3 vertices.");

        //std::vector<uint32_t> faceIndices(count);
        //for (int k = 0; k < count; ++k) {
            //ss >> faceIndices[k];
        //}

        //massert(!ss.fail(), "Error: Face index parsing failed.");

        //for (int k = 1; k < count - 1; ++k) {
            //faces.push_back(faceIndices[0]);
            //faces.push_back(faceIndices[k]);
            //faces.push_back(faceIndices[k + 1]);
        //}
    //}

    //return true;
//}

}

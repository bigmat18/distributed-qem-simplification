#include "massert.hpp"
#include <cstdint>
#include <fstream>
#include <print>
#include <qem_mesh.hpp>
#include <string>

namespace qems {

bool import_mesh_data(std::filesystem::path path, 
                      std::vector<float>& vertices, 
                      std::vector<uint32_t>& faces) 
{
    vertices.clear();
    faces.clear();

    std::ifstream file(path);
    massert(file.is_open(), "Error: Failed to open PLY file.");

    std::string line;
    size_t vertexCount = 0;
    size_t faceCount = 0;
    bool headerEnded = false;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;


        if (token == "format") {
            std::string formatType;
            ss >> formatType;
            massert(formatType == "ascii", 
                    "Error: Input file is not ASCII. Binary PLY is not supported.");
        } else if (token == "element") {
            ss >> token;
            if (token == "vertex") {
                ss >> vertexCount;
            } else if (token == "face") {
                ss >> faceCount;
            }
        } else if (token == "end_header") {
            headerEnded = true;
            break;
        }
    }

    massert(headerEnded, "Error: Malformed PLY header or missing end_header."); 
    massert(vertexCount > 0, "Error: PLY file does not declare vertices.");

    vertices.reserve(vertexCount * 3);
    faces.reserve(faceCount * 3);

    for (size_t i = 0; i < vertexCount; ++i) {
        massert(!std::getline(file, line).fail(),
                "Error: Unexpected EOF while reading vertices.");
        
        std::stringstream ss(line);
        float x, y, z;
        ss >> x >> y >> z;

        massert(!ss.fail(), "Error: Vertex parsing failed (non-numeric data?).");

        vertices.push_back(x);
        vertices.push_back(y);
        vertices.push_back(z);
        std::println("{} {} {}", x,y,z);
    }

    for (size_t i = 0; i < faceCount; ++i) {
        massert(!std::getline(file, line).fail(),
                "Error: Unexpected EOF while reading faces.");
        
        std::stringstream ss(line);
        int count;
        ss >> count;

        massert(count >= 3, "Error: Degenerate face found with < 3 vertices.");

        std::vector<uint32_t> faceIndices(count);
        for (int k = 0; k < count; ++k) {
            ss >> faceIndices[k];
        }

        massert(!ss.fail(), "Error: Face index parsing failed.");

        for (int k = 1; k < count - 1; ++k) {
            faces.push_back(faceIndices[0]);
            faces.push_back(faceIndices[k]);
            faces.push_back(faceIndices[k + 1]);
        }
    }

    return true;
}

bool import_qem_mesh(std::filesystem::path path, 
                     QEMMesh& mesh)
{
    massert(false, "Not implemented yet");
    return true;
}
}

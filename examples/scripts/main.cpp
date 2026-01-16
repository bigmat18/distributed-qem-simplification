#include "logging.hpp"
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
#include <vector>

#include <utils.hpp>
#include <qem_mesh.hpp>
#include <mesh_import.hpp>

int main(int argc, char **argv) {
    omlog().disable();
    omout().disable();
    omerr().disable();

    cxxopts::Options options("cli", "CLI app to test distributed mesh simplification");
    options.add_options()      
        ("i,input", "Input Folder", cxxopts::value<std::string>())
        ("o,output", "Output Folder", cxxopts::value<std::string>());

    options.parse_positional({"input"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    const std::string INPUT  = result["input"].as<std::string>();
    const std::string OUTPUT = result["output"].as<std::string>();

    massert(fs::exists(INPUT) && fs::is_directory(INPUT), 
            "Input must be a valid folder");
    massert(fs::exists(OUTPUT) && fs::is_directory(OUTPUT), 
            "Output must be a valid folder");

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(INPUT)) {
        if (!entry.is_directory()) 
            continue;
    
        fs::path subdir = entry.path();
        fs::path obj_path;
    
        for (const auto& f : fs::directory_iterator(subdir))
            if (f.is_regular_file() && f.path().extension() == ".obj")
                obj_path = f.path();
                break;
    
        if (!obj_path.empty()) 
            files.push_back(obj_path);
    }

    const fs::path output_path(OUTPUT); 
    for (std::size_t i = 0; i < files.size(); i++) {
        qems::QEMMesh mesh;
        const auto file = files[i];
        const auto name = file.parent_path().filename();

        fs::path out_file = output_path / name;
        out_file.replace_extension(".ply");

        massert(OpenMesh::IO::read_mesh(mesh, file.string()), 
                "Error in mesh import " + file.string());
        massert(OpenMesh::IO::write_mesh(mesh, out_file.string()),
                "Error in mesh export " + out_file.string());

        LOG_INFO("{}", out_file.string());
    }

    return 0;
}

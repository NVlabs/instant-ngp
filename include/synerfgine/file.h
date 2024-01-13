#pragma once
#include <filesystem>

namespace fs = std::filesystem;

namespace sng {

struct Utils {
public:
    static inline fs::path get_root_dir() {
        if (!root_dir.empty()) {
            return root_dir;
        }
        root_dir = fs::current_path();
        fs::path exists_in_root_dir = "scripts";
        for (const auto& candidate : {
            fs::path{"."}/exists_in_root_dir,
            fs::path{".."}/exists_in_root_dir,
            root_dir/exists_in_root_dir,
            root_dir/".."/exists_in_root_dir,
        }) {
            if (fs::exists(candidate)) {
                root_dir = candidate.parent_path();
                break;
            }
        }
        return root_dir;
    }

private:
    static fs::path root_dir;
};

fs::path Utils::root_dir{};

}
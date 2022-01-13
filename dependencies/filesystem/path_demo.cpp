#include <iostream>
#include "filesystem/path.h"
#include "filesystem/resolver.h"

using namespace std;
using namespace filesystem;

int main(int argc, char **argv) {
#if !defined(WIN32)
    path path1("/dir 1/dir 2/");
#else
    path path1("C:\\dir 1\\dir 2\\");
#endif
    path path2("dir 3");

    cout << path1.exists() << endl;
    cout << path1 << endl;
    cout << (path1/path2) << endl;
    cout << (path1/path2).parent_path() << endl;
    cout << (path1/path2).parent_path().parent_path() << endl;
    cout << (path1/path2).parent_path().parent_path().parent_path() << endl;
    cout << (path1/path2).parent_path().parent_path().parent_path().parent_path() << endl;
    cout << path().parent_path() << endl;
    cout << "some/path.ext:operator==() = " << (path("some/path.ext") == path("some/path.ext")) << endl;
    cout << "some/path.ext:operator==() (unequal) = " << (path("some/path.ext") == path("another/path.ext")) << endl;

    cout << "nonexistant:exists = " << path("nonexistant").exists() << endl;
    cout << "nonexistant:is_file = " << path("nonexistant").is_file() << endl;
    cout << "nonexistant:is_directory = " << path("nonexistant").is_directory() << endl;
    cout << "nonexistant:filename = " << path("nonexistant").filename() << endl;
    cout << "nonexistant:extension = " << path("nonexistant").extension() << endl;
    cout << "filesystem/path.h:exists = " << path("filesystem/path.h").exists() << endl;
    cout << "filesystem/path.h:is_file = " << path("filesystem/path.h").is_file() << endl;
    cout << "filesystem/path.h:is_directory = " << path("filesystem/path.h").is_directory() << endl;
    cout << "filesystem/path.h:filename = " << path("filesystem/path.h").filename() << endl;
    cout << "filesystem/path.h:extension = " << path("filesystem/path.h").extension() << endl;
    cout << "filesystem/path.h:make_absolute = " << path("filesystem/path.h").make_absolute() << endl;
    cout << "../filesystem:exists = " << path("../filesystem").exists() << endl;
    cout << "../filesystem:is_file = " << path("../filesystem").is_file() << endl;
    cout << "../filesystem:is_directory = " << path("../filesystem").is_directory() << endl;
    cout << "../filesystem:extension = " << path("../filesystem").extension() << endl;
    cout << "../filesystem:filename = " << path("../filesystem").filename() << endl;
    cout << "../filesystem:make_absolute = " << path("../filesystem").make_absolute() << endl;

    cout << "resolve(filesystem/path.h) = " << resolver().resolve("filesystem/path.h") << endl;
    cout << "resolve(nonexistant) = " << resolver().resolve("nonexistant") << endl;
    return 0;
}

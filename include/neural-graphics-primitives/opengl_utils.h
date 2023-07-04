#pragma once
#include <neural-graphics-primitives/common.h>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

NGP_NAMESPACE_BEGIN
#ifdef NGP_GUI

enum eShaderType {
    Fragment,
    Vertex,
    Geometry
};

void glCheckError(const char* file, unsigned int line);
bool check_shader(uint32_t handle, const char* desc, bool program);
uint32_t compile_shader(eShaderType shader_type, const char* code);
#endif //NGP_GUI
NGP_NAMESPACE_END
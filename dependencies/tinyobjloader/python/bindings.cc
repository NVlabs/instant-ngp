#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

// Use double precision for better python integration.
#define TINYOBJLOADER_USE_DOUBLE

// define some helper functions for pybind11
#define TINY_OBJ_LOADER_PYTHON_BINDING
#include "tiny_obj_loader.h"

namespace py = pybind11;

using namespace tinyobj;

PYBIND11_MODULE(tinyobjloader, tobj_module)
{
  tobj_module.doc() = "Python bindings for TinyObjLoader.";

  // register struct
  py::class_<ObjReaderConfig>(tobj_module, "ObjReaderConfig")
    .def(py::init<>())
    .def_readwrite("triangulate", &ObjReaderConfig::triangulate);

  // py::init<>() for default constructor
  py::class_<ObjReader>(tobj_module, "ObjReader")
    .def(py::init<>())
    .def("ParseFromFile", &ObjReader::ParseFromFile, py::arg("filename"), py::arg("option") = ObjReaderConfig())
    .def("ParseFromString", &ObjReader::ParseFromString, py::arg("obj_text"), py::arg("mtl_text"), py::arg("option") = ObjReaderConfig())
    .def("Valid", &ObjReader::Valid)
    .def("GetAttrib", &ObjReader::GetAttrib)
    .def("GetShapes", &ObjReader::GetShapes)
    .def("GetMaterials", &ObjReader::GetMaterials)
    .def("Warning", &ObjReader::Warning)
    .def("Error", &ObjReader::Error);

  py::class_<attrib_t>(tobj_module, "attrib_t")
    .def(py::init<>())
    .def_readonly("vertices", &attrib_t::vertices)
    .def("numpy_vertices", [] (attrib_t &instance) {
        auto ret = py::array_t<real_t>(instance.vertices.size());
        py::buffer_info buf = ret.request();
        memcpy(buf.ptr, instance.vertices.data(), instance.vertices.size() * sizeof(real_t));
        return ret;
    })
    .def_readonly("normals", &attrib_t::normals)
    .def_readonly("texcoords", &attrib_t::texcoords)
    .def_readonly("colors", &attrib_t::colors)
    ;

  py::class_<shape_t>(tobj_module, "shape_t")
    .def(py::init<>())
    .def_readwrite("name", &shape_t::name)
    .def_readwrite("mesh", &shape_t::mesh)
    .def_readwrite("lines", &shape_t::lines)
    .def_readwrite("points", &shape_t::points);

  py::class_<index_t>(tobj_module, "index_t")
    .def(py::init<>())
    .def_readwrite("vertex_index", &index_t::vertex_index)
    .def_readwrite("normal_index", &index_t::normal_index)
    .def_readwrite("texcoord_index", &index_t::texcoord_index)
    ;

  // NOTE(syoyo): It looks it is rather difficult to expose assignment by array index to
  // python world for array variable.
  // For example following python scripting does not work well.
  //
  // print(mat.diffuse)
  // >>> [0.1, 0.2, 0.3]
  // mat.diffuse[1] = 1.0
  // print(mat.diffuse)
  // >>> [0.1, 0.2, 0.3]  # No modification
  //
  // https://github.com/pybind/pybind11/issues/1134
  //
  // so, we need to update array variable like this:
  //
  // diffuse = mat.diffuse
  // diffuse[1] = 1.0
  // mat.diffuse = diffuse
  //
  py::class_<material_t>(tobj_module, "material_t")
    .def(py::init<>())
    .def_readwrite("name", &material_t::name)
    .def_property("ambient", &material_t::GetAmbient, &material_t::SetAmbient)
    .def_property("diffuse", &material_t::GetDiffuse, &material_t::SetDiffuse)
    .def_property("specular", &material_t::GetSpecular, &material_t::SetSpecular)
    .def_property("transmittance", &material_t::GetTransmittance, &material_t::SetTransmittance)
    .def_readwrite("shininess", &material_t::shininess)
    .def_readwrite("ior", &material_t::ior)
    .def_readwrite("dissolve", &material_t::dissolve)
    .def_readwrite("illum", &material_t::illum)
    .def_readwrite("ambient_texname", &material_t::ambient_texname)
    .def_readwrite("diffuse_texname", &material_t::diffuse_texname)
    .def_readwrite("specular_texname", &material_t::specular_texname)
    .def_readwrite("specular_highlight_texname", &material_t::specular_highlight_texname)
    .def_readwrite("bump_texname", &material_t::bump_texname)
    .def_readwrite("displacement_texname", &material_t::displacement_texname)
    .def_readwrite("alpha_texname", &material_t::alpha_texname)
    .def_readwrite("reflection_texname", &material_t::reflection_texname)
    // TODO(syoyo): Expose texture parameter
    // PBR
    .def_readwrite("roughness", &material_t::roughness)
    .def_readwrite("metallic", &material_t::metallic)
    .def_readwrite("sheen", &material_t::sheen)
    .def_readwrite("clearcoat_thickness", &material_t::clearcoat_thickness)
    .def_readwrite("clearcoat_roughness", &material_t::clearcoat_roughness)
    .def_readwrite("anisotropy", &material_t::anisotropy)
    .def_readwrite("anisotropy_rotation", &material_t::anisotropy_rotation)

    .def_readwrite("roughness_texname", &material_t::roughness_texname)
    .def_readwrite("metallic_texname", &material_t::metallic_texname)
    .def_readwrite("sheen_texname", &material_t::sheen_texname)
    .def_readwrite("emissive_texname", &material_t::emissive_texname)
    .def_readwrite("normal_texname", &material_t::normal_texname)

    .def("GetCustomParameter", &material_t::GetCustomParameter)
    ;

  py::class_<mesh_t>(tobj_module, "mesh_t")
    .def(py::init<>())
    .def_readonly("num_face_vertices", &mesh_t::num_face_vertices)
    .def("numpy_num_face_vertices", [] (mesh_t &instance) {
        auto ret = py::array_t<unsigned char>(instance.num_face_vertices.size());
        py::buffer_info buf = ret.request();
        memcpy(buf.ptr, instance.num_face_vertices.data(), instance.num_face_vertices.size() * sizeof(unsigned char));
        return ret;
    })
    .def_readonly("indices", &mesh_t::indices)
    .def("numpy_indices", [] (mesh_t &instance) {
        // Flatten indexes. index_t is composed of 3 ints(vertex_index, normal_index, texcoord_index).
        // numpy_indices = [0, -1, -1, 1, -1, -1, ...]
        // C++11 or later should pack POD struct tightly and does not reorder variables, 
        // so we can memcpy to copy data.
        // Still, we check the size of struct and byte offsets of each variable just for sure.
        static_assert(sizeof(index_t) == 12, "sizeof(index_t) must be 12");
        static_assert(offsetof(index_t, vertex_index) == 0, "offsetof(index_t, vertex_index) must be 0");
        static_assert(offsetof(index_t, normal_index) == 4, "offsetof(index_t, normal_index) must be 4");
        static_assert(offsetof(index_t, texcoord_index) == 8, "offsetof(index_t, texcoord_index) must be 8");

        auto ret = py::array_t<int>(instance.indices.size() * 3);
        py::buffer_info buf = ret.request();
        memcpy(buf.ptr, instance.indices.data(), instance.indices.size() * 3 * sizeof(int));
        return ret;
    })
    .def_readonly("material_ids", &mesh_t::material_ids)
    .def("numpy_material_ids", [] (mesh_t &instance) {
        auto ret = py::array_t<int>(instance.material_ids.size());
        py::buffer_info buf = ret.request();
        memcpy(buf.ptr, instance.material_ids.data(), instance.material_ids.size() * sizeof(int));
        return ret;
    });

  py::class_<lines_t>(tobj_module, "lines_t")
    .def(py::init<>());

  py::class_<points_t>(tobj_module, "points_t")
    .def(py::init<>());

}


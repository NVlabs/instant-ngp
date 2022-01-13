/*
The MIT License (MIT)

Copyright (c) 2012-Present, Syoyo Fujita and many contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//
// version 2.0.0 : Add new object oriented API. 1.x API is still provided.
//                 * Support line primitive.
//                 * Support points primitive.
//                 * Support multiple search path for .mtl(v1 API).
//                 * Support vertex weight `vw`(as an tinyobj extension)
//                 * Support escaped whitespece in mtllib
//                 * Add robust triangulation using Mapbox earcut(TINYOBJLOADER_USE_MAPBOX_EARCUT).
// version 1.4.0 : Modifed ParseTextureNameAndOption API
// version 1.3.1 : Make ParseTextureNameAndOption API public
// version 1.3.0 : Separate warning and error message(breaking API of LoadObj)
// version 1.2.3 : Added color space extension('-colorspace') to tex opts.
// version 1.2.2 : Parse multiple group names.
// version 1.2.1 : Added initial support for line('l') primitive(PR #178)
// version 1.2.0 : Hardened implementation(#175)
// version 1.1.1 : Support smoothing groups(#162)
// version 1.1.0 : Support parsing vertex color(#144)
// version 1.0.8 : Fix parsing `g` tag just after `usemtl`(#138)
// version 1.0.7 : Support multiple tex options(#126)
// version 1.0.6 : Add TINYOBJLOADER_USE_DOUBLE option(#124)
// version 1.0.5 : Ignore `Tr` when `d` exists in MTL(#43)
// version 1.0.4 : Support multiple filenames for 'mtllib'(#112)
// version 1.0.3 : Support parsing texture options(#85)
// version 1.0.2 : Improve parsing speed by about a factor of 2 for large
// files(#105)
// version 1.0.1 : Fixes a shape is lost if obj ends with a 'usemtl'(#104)
// version 1.0.0 : Change data structure. Change license from BSD to MIT.
//

//
// Use this in *one* .cc
//   #define TINYOBJLOADER_IMPLEMENTATION
//   #include "tiny_obj_loader.h"
//

#ifndef TINY_OBJ_LOADER_H_
#define TINY_OBJ_LOADER_H_

#include <map>
#include <string>
#include <vector>

namespace tinyobj {

// TODO(syoyo): Better C++11 detection for older compiler
#if __cplusplus > 199711L
#define TINYOBJ_OVERRIDE override
#else
#define TINYOBJ_OVERRIDE
#endif

#ifdef __clang__
#pragma clang diagnostic push
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#pragma clang diagnostic ignored "-Wpadded"

#endif

// https://en.wikipedia.org/wiki/Wavefront_.obj_file says ...
//
//  -blendu on | off                       # set horizontal texture blending
//  (default on)
//  -blendv on | off                       # set vertical texture blending
//  (default on)
//  -boost real_value                      # boost mip-map sharpness
//  -mm base_value gain_value              # modify texture map values (default
//  0 1)
//                                         #     base_value = brightness,
//                                         gain_value = contrast
//  -o u [v [w]]                           # Origin offset             (default
//  0 0 0)
//  -s u [v [w]]                           # Scale                     (default
//  1 1 1)
//  -t u [v [w]]                           # Turbulence                (default
//  0 0 0)
//  -texres resolution                     # texture resolution to create
//  -clamp on | off                        # only render texels in the clamped
//  0-1 range (default off)
//                                         #   When unclamped, textures are
//                                         repeated across a surface,
//                                         #   when clamped, only texels which
//                                         fall within the 0-1
//                                         #   range are rendered.
//  -bm mult_value                         # bump multiplier (for bump maps
//  only)
//
//  -imfchan r | g | b | m | l | z         # specifies which channel of the file
//  is used to
//                                         # create a scalar or bump texture.
//                                         r:red, g:green,
//                                         # b:blue, m:matte, l:luminance,
//                                         z:z-depth..
//                                         # (the default for bump is 'l' and
//                                         for decal is 'm')
//  bump -imfchan r bumpmap.tga            # says to use the red channel of
//  bumpmap.tga as the bumpmap
//
// For reflection maps...
//
//   -type sphere                           # specifies a sphere for a "refl"
//   reflection map
//   -type cube_top    | cube_bottom |      # when using a cube map, the texture
//   file for each
//         cube_front  | cube_back   |      # side of the cube is specified
//         separately
//         cube_left   | cube_right
//
// TinyObjLoader extension.
//
//   -colorspace SPACE                      # Color space of the texture. e.g.
//   'sRGB` or 'linear'
//

#ifdef TINYOBJLOADER_USE_DOUBLE
//#pragma message "using double"
typedef double real_t;
#else
//#pragma message "using float"
typedef float real_t;
#endif

typedef enum {
  TEXTURE_TYPE_NONE,  // default
  TEXTURE_TYPE_SPHERE,
  TEXTURE_TYPE_CUBE_TOP,
  TEXTURE_TYPE_CUBE_BOTTOM,
  TEXTURE_TYPE_CUBE_FRONT,
  TEXTURE_TYPE_CUBE_BACK,
  TEXTURE_TYPE_CUBE_LEFT,
  TEXTURE_TYPE_CUBE_RIGHT
} texture_type_t;

struct texture_option_t {
  texture_type_t type;      // -type (default TEXTURE_TYPE_NONE)
  real_t sharpness;         // -boost (default 1.0?)
  real_t brightness;        // base_value in -mm option (default 0)
  real_t contrast;          // gain_value in -mm option (default 1)
  real_t origin_offset[3];  // -o u [v [w]] (default 0 0 0)
  real_t scale[3];          // -s u [v [w]] (default 1 1 1)
  real_t turbulence[3];     // -t u [v [w]] (default 0 0 0)
  int texture_resolution;   // -texres resolution (No default value in the spec.
                            // We'll use -1)
  bool clamp;               // -clamp (default false)
  char imfchan;  // -imfchan (the default for bump is 'l' and for decal is 'm')
  bool blendu;   // -blendu (default on)
  bool blendv;   // -blendv (default on)
  real_t bump_multiplier;  // -bm (for bump maps only, default 1.0)

  // extension
  std::string colorspace;  // Explicitly specify color space of stored texel
                           // value. Usually `sRGB` or `linear` (default empty).
};

struct material_t {
  std::string name;

  real_t ambient[3];
  real_t diffuse[3];
  real_t specular[3];
  real_t transmittance[3];
  real_t emission[3];
  real_t shininess;
  real_t ior;       // index of refraction
  real_t dissolve;  // 1 == opaque; 0 == fully transparent
  // illumination model (see http://www.fileformat.info/format/material/)
  int illum;

  int dummy;  // Suppress padding warning.

  std::string ambient_texname;             // map_Ka
  std::string diffuse_texname;             // map_Kd
  std::string specular_texname;            // map_Ks
  std::string specular_highlight_texname;  // map_Ns
  std::string bump_texname;                // map_bump, map_Bump, bump
  std::string displacement_texname;        // disp
  std::string alpha_texname;               // map_d
  std::string reflection_texname;          // refl

  texture_option_t ambient_texopt;
  texture_option_t diffuse_texopt;
  texture_option_t specular_texopt;
  texture_option_t specular_highlight_texopt;
  texture_option_t bump_texopt;
  texture_option_t displacement_texopt;
  texture_option_t alpha_texopt;
  texture_option_t reflection_texopt;

  // PBR extension
  // http://exocortex.com/blog/extending_wavefront_mtl_to_support_pbr
  real_t roughness;            // [0, 1] default 0
  real_t metallic;             // [0, 1] default 0
  real_t sheen;                // [0, 1] default 0
  real_t clearcoat_thickness;  // [0, 1] default 0
  real_t clearcoat_roughness;  // [0, 1] default 0
  real_t anisotropy;           // aniso. [0, 1] default 0
  real_t anisotropy_rotation;  // anisor. [0, 1] default 0
  real_t pad0;
  std::string roughness_texname;  // map_Pr
  std::string metallic_texname;   // map_Pm
  std::string sheen_texname;      // map_Ps
  std::string emissive_texname;   // map_Ke
  std::string normal_texname;     // norm. For normal mapping.

  texture_option_t roughness_texopt;
  texture_option_t metallic_texopt;
  texture_option_t sheen_texopt;
  texture_option_t emissive_texopt;
  texture_option_t normal_texopt;

  int pad2;

  std::map<std::string, std::string> unknown_parameter;

#ifdef TINY_OBJ_LOADER_PYTHON_BINDING
  // For pybind11
  std::array<double, 3> GetDiffuse() {
    std::array<double, 3> values;
    values[0] = double(diffuse[0]);
    values[1] = double(diffuse[1]);
    values[2] = double(diffuse[2]);

    return values;
  }

  std::array<double, 3> GetSpecular() {
    std::array<double, 3> values;
    values[0] = double(specular[0]);
    values[1] = double(specular[1]);
    values[2] = double(specular[2]);

    return values;
  }

  std::array<double, 3> GetTransmittance() {
    std::array<double, 3> values;
    values[0] = double(transmittance[0]);
    values[1] = double(transmittance[1]);
    values[2] = double(transmittance[2]);

    return values;
  }

  std::array<double, 3> GetEmission() {
    std::array<double, 3> values;
    values[0] = double(emission[0]);
    values[1] = double(emission[1]);
    values[2] = double(emission[2]);

    return values;
  }

  std::array<double, 3> GetAmbient() {
    std::array<double, 3> values;
    values[0] = double(ambient[0]);
    values[1] = double(ambient[1]);
    values[2] = double(ambient[2]);

    return values;
  }

  void SetDiffuse(std::array<double, 3> &a) {
    diffuse[0] = real_t(a[0]);
    diffuse[1] = real_t(a[1]);
    diffuse[2] = real_t(a[2]);
  }

  void SetAmbient(std::array<double, 3> &a) {
    ambient[0] = real_t(a[0]);
    ambient[1] = real_t(a[1]);
    ambient[2] = real_t(a[2]);
  }

  void SetSpecular(std::array<double, 3> &a) {
    specular[0] = real_t(a[0]);
    specular[1] = real_t(a[1]);
    specular[2] = real_t(a[2]);
  }

  void SetTransmittance(std::array<double, 3> &a) {
    transmittance[0] = real_t(a[0]);
    transmittance[1] = real_t(a[1]);
    transmittance[2] = real_t(a[2]);
  }

  std::string GetCustomParameter(const std::string &key) {
    std::map<std::string, std::string>::const_iterator it =
        unknown_parameter.find(key);

    if (it != unknown_parameter.end()) {
      return it->second;
    }
    return std::string();
  }

#endif
};

struct tag_t {
  std::string name;

  std::vector<int> intValues;
  std::vector<real_t> floatValues;
  std::vector<std::string> stringValues;
};

struct joint_and_weight_t {
  int joint_id;
  real_t weight;
};

struct skin_weight_t {
  int vertex_id;  // Corresponding vertex index in `attrib_t::vertices`.
                  // Compared to `index_t`, this index must be positive and
                  // start with 0(does not allow relative indexing)
  std::vector<joint_and_weight_t> weightValues;
};

// Index struct to support different indices for vtx/normal/texcoord.
// -1 means not used.
struct index_t {
  int vertex_index;
  int normal_index;
  int texcoord_index;
};

struct mesh_t {
  std::vector<index_t> indices;
  std::vector<unsigned char>
      num_face_vertices;          // The number of vertices per
                                  // face. 3 = triangle, 4 = quad,
                                  // ... Up to 255 vertices per face.
  std::vector<int> material_ids;  // per-face material ID
  std::vector<unsigned int> smoothing_group_ids;  // per-face smoothing group
                                                  // ID(0 = off. positive value
                                                  // = group id)
  std::vector<tag_t> tags;                        // SubD tag
};

// struct path_t {
//  std::vector<int> indices;  // pairs of indices for lines
//};

struct lines_t {
  // Linear flattened indices.
  std::vector<index_t> indices;        // indices for vertices(poly lines)
  std::vector<int> num_line_vertices;  // The number of vertices per line.
};

struct points_t {
  std::vector<index_t> indices;  // indices for points
};

struct shape_t {
  std::string name;
  mesh_t mesh;
  lines_t lines;
  points_t points;
};

// Vertex attributes
struct attrib_t {
  std::vector<real_t> vertices;  // 'v'(xyz)

  // For backward compatibility, we store vertex weight in separate array.
  std::vector<real_t> vertex_weights;  // 'v'(w)
  std::vector<real_t> normals;         // 'vn'
  std::vector<real_t> texcoords;       // 'vt'(uv)

  // For backward compatibility, we store texture coordinate 'w' in separate
  // array.
  std::vector<real_t> texcoord_ws;  // 'vt'(w)
  std::vector<real_t> colors;       // extension: vertex colors

  //
  // TinyObj extension.
  //

  // NOTE(syoyo): array index is based on the appearance order.
  // To get a corresponding skin weight for a specific vertex id `vid`,
  // Need to reconstruct a look up table: `skin_weight_t::vertex_id` == `vid`
  // (e.g. using std::map, std::unordered_map)
  std::vector<skin_weight_t> skin_weights;

  attrib_t() {}

  //
  // For pybind11
  //
  const std::vector<real_t> &GetVertices() const { return vertices; }

  const std::vector<real_t> &GetVertexWeights() const { return vertex_weights; }
};

struct callback_t {
  // W is optional and set to 1 if there is no `w` item in `v` line
  void (*vertex_cb)(void *user_data, real_t x, real_t y, real_t z, real_t w);
  void (*normal_cb)(void *user_data, real_t x, real_t y, real_t z);

  // y and z are optional and set to 0 if there is no `y` and/or `z` item(s) in
  // `vt` line.
  void (*texcoord_cb)(void *user_data, real_t x, real_t y, real_t z);

  // called per 'f' line. num_indices is the number of face indices(e.g. 3 for
  // triangle, 4 for quad)
  // 0 will be passed for undefined index in index_t members.
  void (*index_cb)(void *user_data, index_t *indices, int num_indices);
  // `name` material name, `material_id` = the array index of material_t[]. -1
  // if
  // a material not found in .mtl
  void (*usemtl_cb)(void *user_data, const char *name, int material_id);
  // `materials` = parsed material data.
  void (*mtllib_cb)(void *user_data, const material_t *materials,
                    int num_materials);
  // There may be multiple group names
  void (*group_cb)(void *user_data, const char **names, int num_names);
  void (*object_cb)(void *user_data, const char *name);

  callback_t()
      : vertex_cb(NULL),
        normal_cb(NULL),
        texcoord_cb(NULL),
        index_cb(NULL),
        usemtl_cb(NULL),
        mtllib_cb(NULL),
        group_cb(NULL),
        object_cb(NULL) {}
};

class MaterialReader {
 public:
  MaterialReader() {}
  virtual ~MaterialReader();

  virtual bool operator()(const std::string &matId,
                          std::vector<material_t> *materials,
                          std::map<std::string, int> *matMap, std::string *warn,
                          std::string *err) = 0;
};

///
/// Read .mtl from a file.
///
class MaterialFileReader : public MaterialReader {
 public:
  // Path could contain separator(';' in Windows, ':' in Posix)
  explicit MaterialFileReader(const std::string &mtl_basedir)
      : m_mtlBaseDir(mtl_basedir) {}
  virtual ~MaterialFileReader() TINYOBJ_OVERRIDE {}
  virtual bool operator()(const std::string &matId,
                          std::vector<material_t> *materials,
                          std::map<std::string, int> *matMap, std::string *warn,
                          std::string *err) TINYOBJ_OVERRIDE;

 private:
  std::string m_mtlBaseDir;
};

///
/// Read .mtl from a stream.
///
class MaterialStreamReader : public MaterialReader {
 public:
  explicit MaterialStreamReader(std::istream &inStream)
      : m_inStream(inStream) {}
  virtual ~MaterialStreamReader() TINYOBJ_OVERRIDE {}
  virtual bool operator()(const std::string &matId,
                          std::vector<material_t> *materials,
                          std::map<std::string, int> *matMap, std::string *warn,
                          std::string *err) TINYOBJ_OVERRIDE;

 private:
  std::istream &m_inStream;
};

// v2 API
struct ObjReaderConfig {
  bool triangulate;  // triangulate polygon?

  // Currently not used.
  // "simple" or empty: Create triangle fan
  // "earcut": Use the algorithm based on Ear clipping
  std::string triangulation_method;

  /// Parse vertex color.
  /// If vertex color is not present, its filled with default value.
  /// false = no vertex color
  /// This will increase memory of parsed .obj
  bool vertex_color;

  ///
  /// Search path to .mtl file.
  /// Default = "" = search from the same directory of .obj file.
  /// Valid only when loading .obj from a file.
  ///
  std::string mtl_search_path;

  ObjReaderConfig()
      : triangulate(true), triangulation_method("simple"), vertex_color(true) {}
};

///
/// Wavefront .obj reader class(v2 API)
///
class ObjReader {
 public:
  ObjReader() : valid_(false) {}

  ///
  /// Load .obj and .mtl from a file.
  ///
  /// @param[in] filename wavefront .obj filename
  /// @param[in] config Reader configuration
  ///
  bool ParseFromFile(const std::string &filename,
                     const ObjReaderConfig &config = ObjReaderConfig());

  ///
  /// Parse .obj from a text string.
  /// Need to supply .mtl text string by `mtl_text`.
  /// This function ignores `mtllib` line in .obj text.
  ///
  /// @param[in] obj_text wavefront .obj filename
  /// @param[in] mtl_text wavefront .mtl filename
  /// @param[in] config Reader configuration
  ///
  bool ParseFromString(const std::string &obj_text, const std::string &mtl_text,
                       const ObjReaderConfig &config = ObjReaderConfig());

  ///
  /// .obj was loaded or parsed correctly.
  ///
  bool Valid() const { return valid_; }

  const attrib_t &GetAttrib() const { return attrib_; }

  const std::vector<shape_t> &GetShapes() const { return shapes_; }

  const std::vector<material_t> &GetMaterials() const { return materials_; }

  ///
  /// Warning message(may be filled after `Load` or `Parse`)
  ///
  const std::string &Warning() const { return warning_; }

  ///
  /// Error message(filled when `Load` or `Parse` failed)
  ///
  const std::string &Error() const { return error_; }

 private:
  bool valid_;

  attrib_t attrib_;
  std::vector<shape_t> shapes_;
  std::vector<material_t> materials_;

  std::string warning_;
  std::string error_;
};

/// ==>>========= Legacy v1 API =============================================

/// Loads .obj from a file.
/// 'attrib', 'shapes' and 'materials' will be filled with parsed shape data
/// 'shapes' will be filled with parsed shape data
/// Returns true when loading .obj become success.
/// Returns warning message into `warn`, and error message into `err`
/// 'mtl_basedir' is optional, and used for base directory for .mtl file.
/// In default(`NULL'), .mtl file is searched from an application's working
/// directory.
/// 'triangulate' is optional, and used whether triangulate polygon face in .obj
/// or not.
/// Option 'default_vcols_fallback' specifies whether vertex colors should
/// always be defined, even if no colors are given (fallback to white).
bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, const char *filename,
             const char *mtl_basedir = NULL, bool triangulate = true,
             bool default_vcols_fallback = true);

/// Loads .obj from a file with custom user callback.
/// .mtl is loaded as usual and parsed material_t data will be passed to
/// `callback.mtllib_cb`.
/// Returns true when loading .obj/.mtl become success.
/// Returns warning message into `warn`, and error message into `err`
/// See `examples/callback_api/` for how to use this function.
bool LoadObjWithCallback(std::istream &inStream, const callback_t &callback,
                         void *user_data = NULL,
                         MaterialReader *readMatFn = NULL,
                         std::string *warn = NULL, std::string *err = NULL);

/// Loads object from a std::istream, uses `readMatFn` to retrieve
/// std::istream for materials.
/// Returns true when loading .obj become success.
/// Returns warning and error message into `err`
bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, std::istream *inStream,
             MaterialReader *readMatFn = NULL, bool triangulate = true,
             bool default_vcols_fallback = true);

/// Loads materials into std::map
void LoadMtl(std::map<std::string, int> *material_map,
             std::vector<material_t> *materials, std::istream *inStream,
             std::string *warning, std::string *err);

///
/// Parse texture name and texture option for custom texture parameter through
/// material::unknown_parameter
///
/// @param[out] texname Parsed texture name
/// @param[out] texopt Parsed texopt
/// @param[in] linebuf Input string
///
bool ParseTextureNameAndOption(std::string *texname, texture_option_t *texopt,
                               const char *linebuf);

/// =<<========== Legacy v1 API =============================================

}  // namespace tinyobj

#endif  // TINY_OBJ_LOADER_H_

#ifdef TINYOBJLOADER_IMPLEMENTATION
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

#ifdef TINYOBJLOADER_USE_MAPBOX_EARCUT

#ifdef TINYOBJLOADER_DONOT_INCLUDE_MAPBOX_EARCUT
// Assume earcut.hpp is included outside of tiny_obj_loader.h
#else

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include <array>
#include "mapbox/earcut.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

#endif  // TINYOBJLOADER_USE_MAPBOX_EARCUT

namespace tinyobj {

MaterialReader::~MaterialReader() {}

struct vertex_index_t {
  int v_idx, vt_idx, vn_idx;
  vertex_index_t() : v_idx(-1), vt_idx(-1), vn_idx(-1) {}
  explicit vertex_index_t(int idx) : v_idx(idx), vt_idx(idx), vn_idx(idx) {}
  vertex_index_t(int vidx, int vtidx, int vnidx)
      : v_idx(vidx), vt_idx(vtidx), vn_idx(vnidx) {}
};

// Internal data structure for face representation
// index + smoothing group.
struct face_t {
  unsigned int
      smoothing_group_id;  // smoothing group id. 0 = smoothing groupd is off.
  int pad_;
  std::vector<vertex_index_t> vertex_indices;  // face vertex indices.

  face_t() : smoothing_group_id(0), pad_(0) {}
};

// Internal data structure for line representation
struct __line_t {
  // l v1/vt1 v2/vt2 ...
  // In the specification, line primitrive does not have normal index, but
  // TinyObjLoader allow it
  std::vector<vertex_index_t> vertex_indices;
};

// Internal data structure for points representation
struct __points_t {
  // p v1 v2 ...
  // In the specification, point primitrive does not have normal index and
  // texture coord index, but TinyObjLoader allow it.
  std::vector<vertex_index_t> vertex_indices;
};

struct tag_sizes {
  tag_sizes() : num_ints(0), num_reals(0), num_strings(0) {}
  int num_ints;
  int num_reals;
  int num_strings;
};

struct obj_shape {
  std::vector<real_t> v;
  std::vector<real_t> vn;
  std::vector<real_t> vt;
};

//
// Manages group of primitives(face, line, points, ...)
struct PrimGroup {
  std::vector<face_t> faceGroup;
  std::vector<__line_t> lineGroup;
  std::vector<__points_t> pointsGroup;

  void clear() {
    faceGroup.clear();
    lineGroup.clear();
    pointsGroup.clear();
  }

  bool IsEmpty() const {
    return faceGroup.empty() && lineGroup.empty() && pointsGroup.empty();
  }

  // TODO(syoyo): bspline, surface, ...
};

// See
// http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
static std::istream &safeGetline(std::istream &is, std::string &t) {
  t.clear();

  // The characters in the stream are read one-by-one using a std::streambuf.
  // That is faster than reading them one-by-one using the std::istream.
  // Code that uses streambuf this way must be guarded by a sentry object.
  // The sentry object performs various tasks,
  // such as thread synchronization and updating the stream state.

  std::istream::sentry se(is, true);
  std::streambuf *sb = is.rdbuf();

  if (se) {
    for (;;) {
      int c = sb->sbumpc();
      switch (c) {
        case '\n':
          return is;
        case '\r':
          if (sb->sgetc() == '\n') sb->sbumpc();
          return is;
        case EOF:
          // Also handle the case when the last line has no line ending
          if (t.empty()) is.setstate(std::ios::eofbit);
          return is;
        default:
          t += static_cast<char>(c);
      }
    }
  }

  return is;
}

#define IS_SPACE(x) (((x) == ' ') || ((x) == '\t'))
#define IS_DIGIT(x) \
  (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10))
#define IS_NEW_LINE(x) (((x) == '\r') || ((x) == '\n') || ((x) == '\0'))

// Make index zero-base, and also support relative index.
static inline bool fixIndex(int idx, int n, int *ret) {
  if (!ret) {
    return false;
  }

  if (idx > 0) {
    (*ret) = idx - 1;
    return true;
  }

  if (idx == 0) {
    // zero is not allowed according to the spec.
    return false;
  }

  if (idx < 0) {
    (*ret) = n + idx;  // negative value = relative
    return true;
  }

  return false;  // never reach here.
}

static inline std::string parseString(const char **token) {
  std::string s;
  (*token) += strspn((*token), " \t");
  size_t e = strcspn((*token), " \t\r");
  s = std::string((*token), &(*token)[e]);
  (*token) += e;
  return s;
}

static inline int parseInt(const char **token) {
  (*token) += strspn((*token), " \t");
  int i = atoi((*token));
  (*token) += strcspn((*token), " \t\r");
  return i;
}

// Tries to parse a floating point number located at s.
//
// s_end should be a location in the string where reading should absolutely
// stop. For example at the end of the string, to prevent buffer overflows.
//
// Parses the following EBNF grammar:
//   sign    = "+" | "-" ;
//   END     = ? anything not in digit ?
//   digit   = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
//   integer = [sign] , digit , {digit} ;
//   decimal = integer , ["." , integer] ;
//   float   = ( decimal , END ) | ( decimal , ("E" | "e") , integer , END ) ;
//
//  Valid strings are for example:
//   -0  +3.1417e+2  -0.0E-3  1.0324  -1.41   11e2
//
// If the parsing is a success, result is set to the parsed value and true
// is returned.
//
// The function is greedy and will parse until any of the following happens:
//  - a non-conforming character is encountered.
//  - s_end is reached.
//
// The following situations triggers a failure:
//  - s >= s_end.
//  - parse failure.
//
static bool tryParseDouble(const char *s, const char *s_end, double *result) {
  if (s >= s_end) {
    return false;
  }

  double mantissa = 0.0;
  // This exponent is base 2 rather than 10.
  // However the exponent we parse is supposed to be one of ten,
  // thus we must take care to convert the exponent/and or the
  // mantissa to a * 2^E, where a is the mantissa and E is the
  // exponent.
  // To get the final double we will use ldexp, it requires the
  // exponent to be in base 2.
  int exponent = 0;

  // NOTE: THESE MUST BE DECLARED HERE SINCE WE ARE NOT ALLOWED
  // TO JUMP OVER DEFINITIONS.
  char sign = '+';
  char exp_sign = '+';
  char const *curr = s;

  // How many characters were read in a loop.
  int read = 0;
  // Tells whether a loop terminated due to reaching s_end.
  bool end_not_reached = false;
  bool leading_decimal_dots = false;

  /*
          BEGIN PARSING.
  */

  // Find out what sign we've got.
  if (*curr == '+' || *curr == '-') {
    sign = *curr;
    curr++;
    if ((curr != s_end) && (*curr == '.')) {
      // accept. Somethig like `.7e+2`, `-.5234`
      leading_decimal_dots = true;
    }
  } else if (IS_DIGIT(*curr)) { /* Pass through. */
  } else if (*curr == '.') {
    // accept. Somethig like `.7e+2`, `-.5234`
    leading_decimal_dots = true;
  } else {
    goto fail;
  }

  // Read the integer part.
  end_not_reached = (curr != s_end);
  if (!leading_decimal_dots) {
    while (end_not_reached && IS_DIGIT(*curr)) {
      mantissa *= 10;
      mantissa += static_cast<int>(*curr - 0x30);
      curr++;
      read++;
      end_not_reached = (curr != s_end);
    }

    // We must make sure we actually got something.
    if (read == 0) goto fail;
  }

  // We allow numbers of form "#", "###" etc.
  if (!end_not_reached) goto assemble;

  // Read the decimal part.
  if (*curr == '.') {
    curr++;
    read = 1;
    end_not_reached = (curr != s_end);
    while (end_not_reached && IS_DIGIT(*curr)) {
      static const double pow_lut[] = {
          1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
      };
      const int lut_entries = sizeof pow_lut / sizeof pow_lut[0];

      // NOTE: Don't use powf here, it will absolutely murder precision.
      mantissa += static_cast<int>(*curr - 0x30) *
                  (read < lut_entries ? pow_lut[read] : std::pow(10.0, -read));
      read++;
      curr++;
      end_not_reached = (curr != s_end);
    }
  } else if (*curr == 'e' || *curr == 'E') {
  } else {
    goto assemble;
  }

  if (!end_not_reached) goto assemble;

  // Read the exponent part.
  if (*curr == 'e' || *curr == 'E') {
    curr++;
    // Figure out if a sign is present and if it is.
    end_not_reached = (curr != s_end);
    if (end_not_reached && (*curr == '+' || *curr == '-')) {
      exp_sign = *curr;
      curr++;
    } else if (IS_DIGIT(*curr)) { /* Pass through. */
    } else {
      // Empty E is not allowed.
      goto fail;
    }

    read = 0;
    end_not_reached = (curr != s_end);
    while (end_not_reached && IS_DIGIT(*curr)) {
      // To avoid annoying MSVC's min/max macro definiton,
      // Use hardcoded int max value
      if (exponent > (2147483647/10)) { // 2147483647 = std::numeric_limits<int>::max()
        // Integer overflow
        goto fail;
      }
      exponent *= 10;
      exponent += static_cast<int>(*curr - 0x30);
      curr++;
      read++;
      end_not_reached = (curr != s_end);
    }
    exponent *= (exp_sign == '+' ? 1 : -1);
    if (read == 0) goto fail;
  }

assemble:
  *result = (sign == '+' ? 1 : -1) *
            (exponent ? std::ldexp(mantissa * std::pow(5.0, exponent), exponent)
                      : mantissa);
  return true;
fail:
  return false;
}

static inline real_t parseReal(const char **token, double default_value = 0.0) {
  (*token) += strspn((*token), " \t");
  const char *end = (*token) + strcspn((*token), " \t\r");
  double val = default_value;
  tryParseDouble((*token), end, &val);
  real_t f = static_cast<real_t>(val);
  (*token) = end;
  return f;
}

static inline bool parseReal(const char **token, real_t *out) {
  (*token) += strspn((*token), " \t");
  const char *end = (*token) + strcspn((*token), " \t\r");
  double val;
  bool ret = tryParseDouble((*token), end, &val);
  if (ret) {
    real_t f = static_cast<real_t>(val);
    (*out) = f;
  }
  (*token) = end;
  return ret;
}

static inline void parseReal2(real_t *x, real_t *y, const char **token,
                              const double default_x = 0.0,
                              const double default_y = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
}

static inline void parseReal3(real_t *x, real_t *y, real_t *z,
                              const char **token, const double default_x = 0.0,
                              const double default_y = 0.0,
                              const double default_z = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);
}

static inline void parseV(real_t *x, real_t *y, real_t *z, real_t *w,
                          const char **token, const double default_x = 0.0,
                          const double default_y = 0.0,
                          const double default_z = 0.0,
                          const double default_w = 1.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);
  (*w) = parseReal(token, default_w);
}

// Extension: parse vertex with colors(6 items)
static inline bool parseVertexWithColor(real_t *x, real_t *y, real_t *z,
                                        real_t *r, real_t *g, real_t *b,
                                        const char **token,
                                        const double default_x = 0.0,
                                        const double default_y = 0.0,
                                        const double default_z = 0.0) {
  (*x) = parseReal(token, default_x);
  (*y) = parseReal(token, default_y);
  (*z) = parseReal(token, default_z);

  const bool found_color =
      parseReal(token, r) && parseReal(token, g) && parseReal(token, b);

  if (!found_color) {
    (*r) = (*g) = (*b) = 1.0;
  }

  return found_color;
}

static inline bool parseOnOff(const char **token, bool default_value = true) {
  (*token) += strspn((*token), " \t");
  const char *end = (*token) + strcspn((*token), " \t\r");

  bool ret = default_value;
  if ((0 == strncmp((*token), "on", 2))) {
    ret = true;
  } else if ((0 == strncmp((*token), "off", 3))) {
    ret = false;
  }

  (*token) = end;
  return ret;
}

static inline texture_type_t parseTextureType(
    const char **token, texture_type_t default_value = TEXTURE_TYPE_NONE) {
  (*token) += strspn((*token), " \t");
  const char *end = (*token) + strcspn((*token), " \t\r");
  texture_type_t ty = default_value;

  if ((0 == strncmp((*token), "cube_top", strlen("cube_top")))) {
    ty = TEXTURE_TYPE_CUBE_TOP;
  } else if ((0 == strncmp((*token), "cube_bottom", strlen("cube_bottom")))) {
    ty = TEXTURE_TYPE_CUBE_BOTTOM;
  } else if ((0 == strncmp((*token), "cube_left", strlen("cube_left")))) {
    ty = TEXTURE_TYPE_CUBE_LEFT;
  } else if ((0 == strncmp((*token), "cube_right", strlen("cube_right")))) {
    ty = TEXTURE_TYPE_CUBE_RIGHT;
  } else if ((0 == strncmp((*token), "cube_front", strlen("cube_front")))) {
    ty = TEXTURE_TYPE_CUBE_FRONT;
  } else if ((0 == strncmp((*token), "cube_back", strlen("cube_back")))) {
    ty = TEXTURE_TYPE_CUBE_BACK;
  } else if ((0 == strncmp((*token), "sphere", strlen("sphere")))) {
    ty = TEXTURE_TYPE_SPHERE;
  }

  (*token) = end;
  return ty;
}

static tag_sizes parseTagTriple(const char **token) {
  tag_sizes ts;

  (*token) += strspn((*token), " \t");
  ts.num_ints = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return ts;
  }

  (*token)++;  // Skip '/'

  (*token) += strspn((*token), " \t");
  ts.num_reals = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return ts;
  }
  (*token)++;  // Skip '/'

  ts.num_strings = parseInt(token);

  return ts;
}

// Parse triples with index offsets: i, i/j/k, i//k, i/j
static bool parseTriple(const char **token, int vsize, int vnsize, int vtsize,
                        vertex_index_t *ret) {
  if (!ret) {
    return false;
  }

  vertex_index_t vi(-1);

  if (!fixIndex(atoi((*token)), vsize, &(vi.v_idx))) {
    return false;
  }

  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    (*ret) = vi;
    return true;
  }
  (*token)++;

  // i//k
  if ((*token)[0] == '/') {
    (*token)++;
    if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
      return false;
    }
    (*token) += strcspn((*token), "/ \t\r");
    (*ret) = vi;
    return true;
  }

  // i/j/k or i/j
  if (!fixIndex(atoi((*token)), vtsize, &(vi.vt_idx))) {
    return false;
  }

  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    (*ret) = vi;
    return true;
  }

  // i/j/k
  (*token)++;  // skip '/'
  if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
    return false;
  }
  (*token) += strcspn((*token), "/ \t\r");

  (*ret) = vi;

  return true;
}

// Parse raw triples: i, i/j/k, i//k, i/j
static vertex_index_t parseRawTriple(const char **token) {
  vertex_index_t vi(static_cast<int>(0));  // 0 is an invalid index in OBJ

  vi.v_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return vi;
  }
  (*token)++;

  // i//k
  if ((*token)[0] == '/') {
    (*token)++;
    vi.vn_idx = atoi((*token));
    (*token) += strcspn((*token), "/ \t\r");
    return vi;
  }

  // i/j/k or i/j
  vi.vt_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  if ((*token)[0] != '/') {
    return vi;
  }

  // i/j/k
  (*token)++;  // skip '/'
  vi.vn_idx = atoi((*token));
  (*token) += strcspn((*token), "/ \t\r");
  return vi;
}

bool ParseTextureNameAndOption(std::string *texname, texture_option_t *texopt,
                               const char *linebuf) {
  // @todo { write more robust lexer and parser. }
  bool found_texname = false;
  std::string texture_name;

  const char *token = linebuf;  // Assume line ends with NULL

  while (!IS_NEW_LINE((*token))) {
    token += strspn(token, " \t");  // skip space
    if ((0 == strncmp(token, "-blendu", 7)) && IS_SPACE((token[7]))) {
      token += 8;
      texopt->blendu = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-blendv", 7)) && IS_SPACE((token[7]))) {
      token += 8;
      texopt->blendv = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-clamp", 6)) && IS_SPACE((token[6]))) {
      token += 7;
      texopt->clamp = parseOnOff(&token, /* default */ true);
    } else if ((0 == strncmp(token, "-boost", 6)) && IS_SPACE((token[6]))) {
      token += 7;
      texopt->sharpness = parseReal(&token, 1.0);
    } else if ((0 == strncmp(token, "-bm", 3)) && IS_SPACE((token[3]))) {
      token += 4;
      texopt->bump_multiplier = parseReal(&token, 1.0);
    } else if ((0 == strncmp(token, "-o", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->origin_offset[0]), &(texopt->origin_offset[1]),
                 &(texopt->origin_offset[2]), &token);
    } else if ((0 == strncmp(token, "-s", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->scale[0]), &(texopt->scale[1]), &(texopt->scale[2]),
                 &token, 1.0, 1.0, 1.0);
    } else if ((0 == strncmp(token, "-t", 2)) && IS_SPACE((token[2]))) {
      token += 3;
      parseReal3(&(texopt->turbulence[0]), &(texopt->turbulence[1]),
                 &(texopt->turbulence[2]), &token);
    } else if ((0 == strncmp(token, "-type", 5)) && IS_SPACE((token[5]))) {
      token += 5;
      texopt->type = parseTextureType((&token), TEXTURE_TYPE_NONE);
    } else if ((0 == strncmp(token, "-texres", 7)) && IS_SPACE((token[7]))) {
      token += 7;
      // TODO(syoyo): Check if arg is int type.
      texopt->texture_resolution = parseInt(&token);
    } else if ((0 == strncmp(token, "-imfchan", 8)) && IS_SPACE((token[8]))) {
      token += 9;
      token += strspn(token, " \t");
      const char *end = token + strcspn(token, " \t\r");
      if ((end - token) == 1) {  // Assume one char for -imfchan
        texopt->imfchan = (*token);
      }
      token = end;
    } else if ((0 == strncmp(token, "-mm", 3)) && IS_SPACE((token[3]))) {
      token += 4;
      parseReal2(&(texopt->brightness), &(texopt->contrast), &token, 0.0, 1.0);
    } else if ((0 == strncmp(token, "-colorspace", 11)) &&
               IS_SPACE((token[11]))) {
      token += 12;
      texopt->colorspace = parseString(&token);
    } else {
// Assume texture filename
#if 0
      size_t len = strcspn(token, " \t\r");  // untile next space
      texture_name = std::string(token, token + len);
      token += len;

      token += strspn(token, " \t");  // skip space
#else
      // Read filename until line end to parse filename containing whitespace
      // TODO(syoyo): Support parsing texture option flag after the filename.
      texture_name = std::string(token);
      token += texture_name.length();
#endif

      found_texname = true;
    }
  }

  if (found_texname) {
    (*texname) = texture_name;
    return true;
  } else {
    return false;
  }
}

static void InitTexOpt(texture_option_t *texopt, const bool is_bump) {
  if (is_bump) {
    texopt->imfchan = 'l';
  } else {
    texopt->imfchan = 'm';
  }
  texopt->bump_multiplier = static_cast<real_t>(1.0);
  texopt->clamp = false;
  texopt->blendu = true;
  texopt->blendv = true;
  texopt->sharpness = static_cast<real_t>(1.0);
  texopt->brightness = static_cast<real_t>(0.0);
  texopt->contrast = static_cast<real_t>(1.0);
  texopt->origin_offset[0] = static_cast<real_t>(0.0);
  texopt->origin_offset[1] = static_cast<real_t>(0.0);
  texopt->origin_offset[2] = static_cast<real_t>(0.0);
  texopt->scale[0] = static_cast<real_t>(1.0);
  texopt->scale[1] = static_cast<real_t>(1.0);
  texopt->scale[2] = static_cast<real_t>(1.0);
  texopt->turbulence[0] = static_cast<real_t>(0.0);
  texopt->turbulence[1] = static_cast<real_t>(0.0);
  texopt->turbulence[2] = static_cast<real_t>(0.0);
  texopt->texture_resolution = -1;
  texopt->type = TEXTURE_TYPE_NONE;
}

static void InitMaterial(material_t *material) {
  InitTexOpt(&material->ambient_texopt, /* is_bump */ false);
  InitTexOpt(&material->diffuse_texopt, /* is_bump */ false);
  InitTexOpt(&material->specular_texopt, /* is_bump */ false);
  InitTexOpt(&material->specular_highlight_texopt, /* is_bump */ false);
  InitTexOpt(&material->bump_texopt, /* is_bump */ true);
  InitTexOpt(&material->displacement_texopt, /* is_bump */ false);
  InitTexOpt(&material->alpha_texopt, /* is_bump */ false);
  InitTexOpt(&material->reflection_texopt, /* is_bump */ false);
  InitTexOpt(&material->roughness_texopt, /* is_bump */ false);
  InitTexOpt(&material->metallic_texopt, /* is_bump */ false);
  InitTexOpt(&material->sheen_texopt, /* is_bump */ false);
  InitTexOpt(&material->emissive_texopt, /* is_bump */ false);
  InitTexOpt(&material->normal_texopt,
             /* is_bump */ false);  // @fixme { is_bump will be true? }
  material->name = "";
  material->ambient_texname = "";
  material->diffuse_texname = "";
  material->specular_texname = "";
  material->specular_highlight_texname = "";
  material->bump_texname = "";
  material->displacement_texname = "";
  material->reflection_texname = "";
  material->alpha_texname = "";
  for (int i = 0; i < 3; i++) {
    material->ambient[i] = static_cast<real_t>(0.0);
    material->diffuse[i] = static_cast<real_t>(0.0);
    material->specular[i] = static_cast<real_t>(0.0);
    material->transmittance[i] = static_cast<real_t>(0.0);
    material->emission[i] = static_cast<real_t>(0.0);
  }
  material->illum = 0;
  material->dissolve = static_cast<real_t>(1.0);
  material->shininess = static_cast<real_t>(1.0);
  material->ior = static_cast<real_t>(1.0);

  material->roughness = static_cast<real_t>(0.0);
  material->metallic = static_cast<real_t>(0.0);
  material->sheen = static_cast<real_t>(0.0);
  material->clearcoat_thickness = static_cast<real_t>(0.0);
  material->clearcoat_roughness = static_cast<real_t>(0.0);
  material->anisotropy_rotation = static_cast<real_t>(0.0);
  material->anisotropy = static_cast<real_t>(0.0);
  material->roughness_texname = "";
  material->metallic_texname = "";
  material->sheen_texname = "";
  material->emissive_texname = "";
  material->normal_texname = "";

  material->unknown_parameter.clear();
}

// code from https://wrf.ecse.rpi.edu//Research/Short_Notes/pnpoly.html
template <typename T>
static int pnpoly(int nvert, T *vertx, T *verty, T testx, T testy) {
  int i, j, c = 0;
  for (i = 0, j = nvert - 1; i < nvert; j = i++) {
    if (((verty[i] > testy) != (verty[j] > testy)) &&
        (testx <
         (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) +
             vertx[i]))
      c = !c;
  }
  return c;
}

// TODO(syoyo): refactor function.
static bool exportGroupsToShape(shape_t *shape, const PrimGroup &prim_group,
                                const std::vector<tag_t> &tags,
                                const int material_id, const std::string &name,
                                bool triangulate, const std::vector<real_t> &v,
                                std::string *warn) {
  if (prim_group.IsEmpty()) {
    return false;
  }

  shape->name = name;

  // polygon
  if (!prim_group.faceGroup.empty()) {
    // Flatten vertices and indices
    for (size_t i = 0; i < prim_group.faceGroup.size(); i++) {
      const face_t &face = prim_group.faceGroup[i];

      size_t npolys = face.vertex_indices.size();

      if (npolys < 3) {
        // Face must have 3+ vertices.
        if (warn) {
          (*warn) += "Degenerated face found\n.";
        }
        continue;
      }

      if (triangulate) {
        if (npolys == 4) {
          vertex_index_t i0 = face.vertex_indices[0];
          vertex_index_t i1 = face.vertex_indices[1];
          vertex_index_t i2 = face.vertex_indices[2];
          vertex_index_t i3 = face.vertex_indices[3];

          size_t vi0 = size_t(i0.v_idx);
          size_t vi1 = size_t(i1.v_idx);
          size_t vi2 = size_t(i2.v_idx);
          size_t vi3 = size_t(i3.v_idx);

          if (((3 * vi0 + 2) >= v.size()) || ((3 * vi1 + 2) >= v.size()) ||
              ((3 * vi2 + 2) >= v.size()) || ((3 * vi3 + 2) >= v.size())) {
            // Invalid triangle.
            // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
            if (warn) {
              (*warn) += "Face with invalid vertex index found.\n";
            }
            continue;
          }

          real_t v0x = v[vi0 * 3 + 0];
          real_t v0y = v[vi0 * 3 + 1];
          real_t v0z = v[vi0 * 3 + 2];
          real_t v1x = v[vi1 * 3 + 0];
          real_t v1y = v[vi1 * 3 + 1];
          real_t v1z = v[vi1 * 3 + 2];
          real_t v2x = v[vi2 * 3 + 0];
          real_t v2y = v[vi2 * 3 + 1];
          real_t v2z = v[vi2 * 3 + 2];
          real_t v3x = v[vi3 * 3 + 0];
          real_t v3y = v[vi3 * 3 + 1];
          real_t v3z = v[vi3 * 3 + 2];

          // There are two candidates to split the quad into two triangles.
          //
          // Choose the shortest edge.
          // TODO: Is it better to determine the edge to split by calculating
          // the area of each triangle?
          //
          // +---+
          // |\  |
          // | \ |
          // |  \|
          // +---+
          //
          // +---+
          // |  /|
          // | / |
          // |/  |
          // +---+

          real_t e02x = v2x - v0x;
          real_t e02y = v2y - v0y;
          real_t e02z = v2z - v0z;
          real_t e13x = v3x - v1x;
          real_t e13y = v3y - v1y;
          real_t e13z = v3z - v1z;

          real_t sqr02 = e02x * e02x + e02y * e02y + e02z * e02z;
          real_t sqr13 = e13x * e13x + e13y * e13y + e13z * e13z;

          index_t idx0, idx1, idx2, idx3;

          idx0.vertex_index = i0.v_idx;
          idx0.normal_index = i0.vn_idx;
          idx0.texcoord_index = i0.vt_idx;
          idx1.vertex_index = i1.v_idx;
          idx1.normal_index = i1.vn_idx;
          idx1.texcoord_index = i1.vt_idx;
          idx2.vertex_index = i2.v_idx;
          idx2.normal_index = i2.vn_idx;
          idx2.texcoord_index = i2.vt_idx;
          idx3.vertex_index = i3.v_idx;
          idx3.normal_index = i3.vn_idx;
          idx3.texcoord_index = i3.vt_idx;

          if (sqr02 < sqr13) {
            // [0, 1, 2], [0, 2, 3]
            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx2);

            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx2);
            shape->mesh.indices.push_back(idx3);
          } else {
            // [0, 1, 3], [1, 2, 3]
            shape->mesh.indices.push_back(idx0);
            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx3);

            shape->mesh.indices.push_back(idx1);
            shape->mesh.indices.push_back(idx2);
            shape->mesh.indices.push_back(idx3);
          }

          // Two triangle faces
          shape->mesh.num_face_vertices.push_back(3);
          shape->mesh.num_face_vertices.push_back(3);

          shape->mesh.material_ids.push_back(material_id);
          shape->mesh.material_ids.push_back(material_id);

          shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);
          shape->mesh.smoothing_group_ids.push_back(face.smoothing_group_id);

        } else {
          vertex_index_t i0 = face.vertex_indices[0];
          vertex_index_t i1(-1);
          vertex_index_t i2 = face.vertex_indices[1];

          // find the two axes to work in
          size_t axes[2] = {1, 2};
          for (size_t k = 0; k < npolys; ++k) {
            i0 = face.vertex_indices[(k + 0) % npolys];
            i1 = face.vertex_indices[(k + 1) % npolys];
            i2 = face.vertex_indices[(k + 2) % npolys];
            size_t vi0 = size_t(i0.v_idx);
            size_t vi1 = size_t(i1.v_idx);
            size_t vi2 = size_t(i2.v_idx);

            if (((3 * vi0 + 2) >= v.size()) || ((3 * vi1 + 2) >= v.size()) ||
                ((3 * vi2 + 2) >= v.size())) {
              // Invalid triangle.
              // FIXME(syoyo): Is it ok to simply skip this invalid triangle?
              continue;
            }
            real_t v0x = v[vi0 * 3 + 0];
            real_t v0y = v[vi0 * 3 + 1];
            real_t v0z = v[vi0 * 3 + 2];
            real_t v1x = v[vi1 * 3 + 0];
            real_t v1y = v[vi1 * 3 + 1];
            real_t v1z = v[vi1 * 3 + 2];
            real_t v2x = v[vi2 * 3 + 0];
            real_t v2y = v[vi2 * 3 + 1];
            real_t v2z = v[vi2 * 3 + 2];
            real_t e0x = v1x - v0x;
            real_t e0y = v1y - v0y;
            real_t e0z = v1z - v0z;
            real_t e1x = v2x - v1x;
            real_t e1y = v2y - v1y;
            real_t e1z = v2z - v1z;
            real_t cx = std::fabs(e0y * e1z - e0z * e1y);
            real_t cy = std::fabs(e0z * e1x - e0x * e1z);
            real_t cz = std::fabs(e0x * e1y - e0y * e1x);
            const real_t epsilon = std::numeric_limits<real_t>::epsilon();
            // std::cout << "cx " << cx << ", cy " << cy << ", cz " << cz <<
            // "\n";
            if (cx > epsilon || cy > epsilon || cz > epsilon) {
              // std::cout << "corner\n";
              // found a corner
              if (cx > cy && cx > cz) {
                // std::cout << "pattern0\n";
              } else {
                // std::cout << "axes[0] = 0\n";
                axes[0] = 0;
                if (cz > cx && cz > cy) {
                  // std::cout << "axes[1] = 1\n";
                  axes[1] = 1;
                }
              }
              break;
            }
          }

#ifdef TINYOBJLOADER_USE_MAPBOX_EARCUT
          using Point = std::array<real_t, 2>;

          // first polyline define the main polygon.
          // following polylines define holes(not used in tinyobj).
          std::vector<std::vector<Point> > polygon;

          std::vector<Point> polyline;

          // Fill polygon data(facevarying vertices).
          for (size_t k = 0; k < npolys; k++) {
            i0 = face.vertex_indices[k];
            size_t vi0 = size_t(i0.v_idx);

            assert(((3 * vi0 + 2) < v.size()));

            real_t v0x = v[vi0 * 3 + axes[0]];
            real_t v0y = v[vi0 * 3 + axes[1]];

            polyline.push_back({v0x, v0y});
          }

          polygon.push_back(polyline);
          std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon);
          // => result = 3 * faces, clockwise

          assert(indices.size() % 3 == 0);

          // Reconstruct vertex_index_t
          for (size_t k = 0; k < indices.size() / 3; k++) {
            {
              index_t idx0, idx1, idx2;
              idx0.vertex_index = face.vertex_indices[indices[3 * k + 0]].v_idx;
              idx0.normal_index =
                  face.vertex_indices[indices[3 * k + 0]].vn_idx;
              idx0.texcoord_index =
                  face.vertex_indices[indices[3 * k + 0]].vt_idx;
              idx1.vertex_index = face.vertex_indices[indices[3 * k + 1]].v_idx;
              idx1.normal_index =
                  face.vertex_indices[indices[3 * k + 1]].vn_idx;
              idx1.texcoord_index =
                  face.vertex_indices[indices[3 * k + 1]].vt_idx;
              idx2.vertex_index = face.vertex_indices[indices[3 * k + 2]].v_idx;
              idx2.normal_index =
                  face.vertex_indices[indices[3 * k + 2]].vn_idx;
              idx2.texcoord_index =
                  face.vertex_indices[indices[3 * k + 2]].vt_idx;

              shape->mesh.indices.push_back(idx0);
              shape->mesh.indices.push_back(idx1);
              shape->mesh.indices.push_back(idx2);

              shape->mesh.num_face_vertices.push_back(3);
              shape->mesh.material_ids.push_back(material_id);
              shape->mesh.smoothing_group_ids.push_back(
                  face.smoothing_group_id);
            }
          }

#else  // Built-in ear clipping triangulation


          face_t remainingFace = face;  // copy
          size_t guess_vert = 0;
          vertex_index_t ind[3];
          real_t vx[3];
          real_t vy[3];

          // How many iterations can we do without decreasing the remaining
          // vertices.
          size_t remainingIterations = face.vertex_indices.size();
          size_t previousRemainingVertices =
              remainingFace.vertex_indices.size();

          while (remainingFace.vertex_indices.size() > 3 &&
                 remainingIterations > 0) {
            // std::cout << "remainingIterations " << remainingIterations <<
            // "\n";

            npolys = remainingFace.vertex_indices.size();
            if (guess_vert >= npolys) {
              guess_vert -= npolys;
            }

            if (previousRemainingVertices != npolys) {
              // The number of remaining vertices decreased. Reset counters.
              previousRemainingVertices = npolys;
              remainingIterations = npolys;
            } else {
              // We didn't consume a vertex on previous iteration, reduce the
              // available iterations.
              remainingIterations--;
            }

            for (size_t k = 0; k < 3; k++) {
              ind[k] = remainingFace.vertex_indices[(guess_vert + k) % npolys];
              size_t vi = size_t(ind[k].v_idx);
              if (((vi * 3 + axes[0]) >= v.size()) ||
                  ((vi * 3 + axes[1]) >= v.size())) {
                // ???
                vx[k] = static_cast<real_t>(0.0);
                vy[k] = static_cast<real_t>(0.0);
              } else {
                vx[k] = v[vi * 3 + axes[0]];
                vy[k] = v[vi * 3 + axes[1]];
              }
            }

            //
            // area is calculated per face
            //
            real_t e0x = vx[1] - vx[0];
            real_t e0y = vy[1] - vy[0];
            real_t e1x = vx[2] - vx[1];
            real_t e1y = vy[2] - vy[1];
            real_t cross = e0x * e1y - e0y * e1x;
            // std::cout << "axes = " << axes[0] << ", " << axes[1] << "\n";
            // std::cout << "e0x, e0y, e1x, e1y " << e0x << ", " << e0y << ", "
            // << e1x << ", " << e1y << "\n";

            real_t area = (vx[0] * vy[1] - vy[0] * vx[1]) * static_cast<real_t>(0.5);
            // std::cout << "cross " << cross << ", area " << area << "\n";
            // if an internal angle
            if (cross * area < static_cast<real_t>(0.0)) {
              // std::cout << "internal \n";
              guess_vert += 1;
              // std::cout << "guess vert : " << guess_vert << "\n";
              continue;
            }

            // check all other verts in case they are inside this triangle
            bool overlap = false;
            for (size_t otherVert = 3; otherVert < npolys; ++otherVert) {
              size_t idx = (guess_vert + otherVert) % npolys;

              if (idx >= remainingFace.vertex_indices.size()) {
                // std::cout << "???0\n";
                // ???
                continue;
              }

              size_t ovi = size_t(remainingFace.vertex_indices[idx].v_idx);

              if (((ovi * 3 + axes[0]) >= v.size()) ||
                  ((ovi * 3 + axes[1]) >= v.size())) {
                // std::cout << "???1\n";
                // ???
                continue;
              }
              real_t tx = v[ovi * 3 + axes[0]];
              real_t ty = v[ovi * 3 + axes[1]];
              if (pnpoly(3, vx, vy, tx, ty)) {
                // std::cout << "overlap\n";
                overlap = true;
                break;
              }
            }

            if (overlap) {
              // std::cout << "overlap2\n";
              guess_vert += 1;
              continue;
            }

            // this triangle is an ear
            {
              index_t idx0, idx1, idx2;
              idx0.vertex_index = ind[0].v_idx;
              idx0.normal_index = ind[0].vn_idx;
              idx0.texcoord_index = ind[0].vt_idx;
              idx1.vertex_index = ind[1].v_idx;
              idx1.normal_index = ind[1].vn_idx;
              idx1.texcoord_index = ind[1].vt_idx;
              idx2.vertex_index = ind[2].v_idx;
              idx2.normal_index = ind[2].vn_idx;
              idx2.texcoord_index = ind[2].vt_idx;

              shape->mesh.indices.push_back(idx0);
              shape->mesh.indices.push_back(idx1);
              shape->mesh.indices.push_back(idx2);

              shape->mesh.num_face_vertices.push_back(3);
              shape->mesh.material_ids.push_back(material_id);
              shape->mesh.smoothing_group_ids.push_back(
                  face.smoothing_group_id);
            }

            // remove v1 from the list
            size_t removed_vert_index = (guess_vert + 1) % npolys;
            while (removed_vert_index + 1 < npolys) {
              remainingFace.vertex_indices[removed_vert_index] =
                  remainingFace.vertex_indices[removed_vert_index + 1];
              removed_vert_index += 1;
            }
            remainingFace.vertex_indices.pop_back();
          }

          // std::cout << "remainingFace.vi.size = " <<
          // remainingFace.vertex_indices.size() << "\n";
          if (remainingFace.vertex_indices.size() == 3) {
            i0 = remainingFace.vertex_indices[0];
            i1 = remainingFace.vertex_indices[1];
            i2 = remainingFace.vertex_indices[2];
            {
              index_t idx0, idx1, idx2;
              idx0.vertex_index = i0.v_idx;
              idx0.normal_index = i0.vn_idx;
              idx0.texcoord_index = i0.vt_idx;
              idx1.vertex_index = i1.v_idx;
              idx1.normal_index = i1.vn_idx;
              idx1.texcoord_index = i1.vt_idx;
              idx2.vertex_index = i2.v_idx;
              idx2.normal_index = i2.vn_idx;
              idx2.texcoord_index = i2.vt_idx;

              shape->mesh.indices.push_back(idx0);
              shape->mesh.indices.push_back(idx1);
              shape->mesh.indices.push_back(idx2);

              shape->mesh.num_face_vertices.push_back(3);
              shape->mesh.material_ids.push_back(material_id);
              shape->mesh.smoothing_group_ids.push_back(
                  face.smoothing_group_id);
            }
          }
#endif
        }  // npolys
      } else {
        for (size_t k = 0; k < npolys; k++) {
          index_t idx;
          idx.vertex_index = face.vertex_indices[k].v_idx;
          idx.normal_index = face.vertex_indices[k].vn_idx;
          idx.texcoord_index = face.vertex_indices[k].vt_idx;
          shape->mesh.indices.push_back(idx);
        }

        shape->mesh.num_face_vertices.push_back(
            static_cast<unsigned char>(npolys));
        shape->mesh.material_ids.push_back(material_id);  // per face
        shape->mesh.smoothing_group_ids.push_back(
            face.smoothing_group_id);  // per face
      }
    }

    shape->mesh.tags = tags;
  }

  // line
  if (!prim_group.lineGroup.empty()) {
    // Flatten indices
    for (size_t i = 0; i < prim_group.lineGroup.size(); i++) {
      for (size_t j = 0; j < prim_group.lineGroup[i].vertex_indices.size();
           j++) {
        const vertex_index_t &vi = prim_group.lineGroup[i].vertex_indices[j];

        index_t idx;
        idx.vertex_index = vi.v_idx;
        idx.normal_index = vi.vn_idx;
        idx.texcoord_index = vi.vt_idx;

        shape->lines.indices.push_back(idx);
      }

      shape->lines.num_line_vertices.push_back(
          int(prim_group.lineGroup[i].vertex_indices.size()));
    }
  }

  // points
  if (!prim_group.pointsGroup.empty()) {
    // Flatten & convert indices
    for (size_t i = 0; i < prim_group.pointsGroup.size(); i++) {
      for (size_t j = 0; j < prim_group.pointsGroup[i].vertex_indices.size();
           j++) {
        const vertex_index_t &vi = prim_group.pointsGroup[i].vertex_indices[j];

        index_t idx;
        idx.vertex_index = vi.v_idx;
        idx.normal_index = vi.vn_idx;
        idx.texcoord_index = vi.vt_idx;

        shape->points.indices.push_back(idx);
      }
    }
  }

  return true;
}

// Split a string with specified delimiter character and escape character.
// https://rosettacode.org/wiki/Tokenize_a_string_with_escaping#C.2B.2B
static void SplitString(const std::string &s, char delim, char escape,
                        std::vector<std::string> &elems) {
  std::string token;

  bool escaping = false;
  for (size_t i = 0; i < s.size(); ++i) {
    char ch = s[i];
    if (escaping) {
      escaping = false;
    } else if (ch == escape) {
      escaping = true;
      continue;
    } else if (ch == delim) {
      if (!token.empty()) {
        elems.push_back(token);
      }
      token.clear();
      continue;
    }
    token += ch;
  }

  elems.push_back(token);
}

static std::string JoinPath(const std::string &dir,
                            const std::string &filename) {
  if (dir.empty()) {
    return filename;
  } else {
    // check '/'
    char lastChar = *dir.rbegin();
    if (lastChar != '/') {
      return dir + std::string("/") + filename;
    } else {
      return dir + filename;
    }
  }
}

void LoadMtl(std::map<std::string, int> *material_map,
             std::vector<material_t> *materials, std::istream *inStream,
             std::string *warning, std::string *err) {
  (void)err;

  // Create a default material anyway.
  material_t material;
  InitMaterial(&material);

  // Issue 43. `d` wins against `Tr` since `Tr` is not in the MTL specification.
  bool has_d = false;
  bool has_tr = false;

  // has_kd is used to set a default diffuse value when map_Kd is present
  // and Kd is not.
  bool has_kd = false;

  std::stringstream warn_ss;

  size_t line_no = 0;
  std::string linebuf;
  while (inStream->peek() != -1) {
    safeGetline(*inStream, linebuf);
    line_no++;

    // Trim trailing whitespace.
    if (linebuf.size() > 0) {
      linebuf = linebuf.substr(0, linebuf.find_last_not_of(" \t") + 1);
    }

    // Trim newline '\r\n' or '\n'
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\n')
        linebuf.erase(linebuf.size() - 1);
    }
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\r')
        linebuf.erase(linebuf.size() - 1);
    }

    // Skip if empty line.
    if (linebuf.empty()) {
      continue;
    }

    // Skip leading space.
    const char *token = linebuf.c_str();
    token += strspn(token, " \t");

    assert(token);
    if (token[0] == '\0') continue;  // empty line

    if (token[0] == '#') continue;  // comment line

    // new mtl
    if ((0 == strncmp(token, "newmtl", 6)) && IS_SPACE((token[6]))) {
      // flush previous material.
      if (!material.name.empty()) {
        material_map->insert(std::pair<std::string, int>(
            material.name, static_cast<int>(materials->size())));
        materials->push_back(material);
      }

      // initial temporary material
      InitMaterial(&material);

      has_d = false;
      has_tr = false;

      // set new mtl name
      token += 7;
      {
        std::stringstream sstr;
        sstr << token;
        material.name = sstr.str();
      }
      continue;
    }

    // ambient
    if (token[0] == 'K' && token[1] == 'a' && IS_SPACE((token[2]))) {
      token += 2;
      real_t r, g, b;
      parseReal3(&r, &g, &b, &token);
      material.ambient[0] = r;
      material.ambient[1] = g;
      material.ambient[2] = b;
      continue;
    }

    // diffuse
    if (token[0] == 'K' && token[1] == 'd' && IS_SPACE((token[2]))) {
      token += 2;
      real_t r, g, b;
      parseReal3(&r, &g, &b, &token);
      material.diffuse[0] = r;
      material.diffuse[1] = g;
      material.diffuse[2] = b;
      has_kd = true;
      continue;
    }

    // specular
    if (token[0] == 'K' && token[1] == 's' && IS_SPACE((token[2]))) {
      token += 2;
      real_t r, g, b;
      parseReal3(&r, &g, &b, &token);
      material.specular[0] = r;
      material.specular[1] = g;
      material.specular[2] = b;
      continue;
    }

    // transmittance
    if ((token[0] == 'K' && token[1] == 't' && IS_SPACE((token[2]))) ||
        (token[0] == 'T' && token[1] == 'f' && IS_SPACE((token[2])))) {
      token += 2;
      real_t r, g, b;
      parseReal3(&r, &g, &b, &token);
      material.transmittance[0] = r;
      material.transmittance[1] = g;
      material.transmittance[2] = b;
      continue;
    }

    // ior(index of refraction)
    if (token[0] == 'N' && token[1] == 'i' && IS_SPACE((token[2]))) {
      token += 2;
      material.ior = parseReal(&token);
      continue;
    }

    // emission
    if (token[0] == 'K' && token[1] == 'e' && IS_SPACE(token[2])) {
      token += 2;
      real_t r, g, b;
      parseReal3(&r, &g, &b, &token);
      material.emission[0] = r;
      material.emission[1] = g;
      material.emission[2] = b;
      continue;
    }

    // shininess
    if (token[0] == 'N' && token[1] == 's' && IS_SPACE(token[2])) {
      token += 2;
      material.shininess = parseReal(&token);
      continue;
    }

    // illum model
    if (0 == strncmp(token, "illum", 5) && IS_SPACE(token[5])) {
      token += 6;
      material.illum = parseInt(&token);
      continue;
    }

    // dissolve
    if ((token[0] == 'd' && IS_SPACE(token[1]))) {
      token += 1;
      material.dissolve = parseReal(&token);

      if (has_tr) {
        warn_ss << "Both `d` and `Tr` parameters defined for \""
                << material.name
                << "\". Use the value of `d` for dissolve (line " << line_no
                << " in .mtl.)\n";
      }
      has_d = true;
      continue;
    }
    if (token[0] == 'T' && token[1] == 'r' && IS_SPACE(token[2])) {
      token += 2;
      if (has_d) {
        // `d` wins. Ignore `Tr` value.
        warn_ss << "Both `d` and `Tr` parameters defined for \""
                << material.name
                << "\". Use the value of `d` for dissolve (line " << line_no
                << " in .mtl.)\n";
      } else {
        // We invert value of Tr(assume Tr is in range [0, 1])
        // NOTE: Interpretation of Tr is application(exporter) dependent. For
        // some application(e.g. 3ds max obj exporter), Tr = d(Issue 43)
        material.dissolve = static_cast<real_t>(1.0) - parseReal(&token);
      }
      has_tr = true;
      continue;
    }

    // PBR: roughness
    if (token[0] == 'P' && token[1] == 'r' && IS_SPACE(token[2])) {
      token += 2;
      material.roughness = parseReal(&token);
      continue;
    }

    // PBR: metallic
    if (token[0] == 'P' && token[1] == 'm' && IS_SPACE(token[2])) {
      token += 2;
      material.metallic = parseReal(&token);
      continue;
    }

    // PBR: sheen
    if (token[0] == 'P' && token[1] == 's' && IS_SPACE(token[2])) {
      token += 2;
      material.sheen = parseReal(&token);
      continue;
    }

    // PBR: clearcoat thickness
    if (token[0] == 'P' && token[1] == 'c' && IS_SPACE(token[2])) {
      token += 2;
      material.clearcoat_thickness = parseReal(&token);
      continue;
    }

    // PBR: clearcoat roughness
    if ((0 == strncmp(token, "Pcr", 3)) && IS_SPACE(token[3])) {
      token += 4;
      material.clearcoat_roughness = parseReal(&token);
      continue;
    }

    // PBR: anisotropy
    if ((0 == strncmp(token, "aniso", 5)) && IS_SPACE(token[5])) {
      token += 6;
      material.anisotropy = parseReal(&token);
      continue;
    }

    // PBR: anisotropy rotation
    if ((0 == strncmp(token, "anisor", 6)) && IS_SPACE(token[6])) {
      token += 7;
      material.anisotropy_rotation = parseReal(&token);
      continue;
    }

    // ambient texture
    if ((0 == strncmp(token, "map_Ka", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.ambient_texname),
                                &(material.ambient_texopt), token);
      continue;
    }

    // diffuse texture
    if ((0 == strncmp(token, "map_Kd", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.diffuse_texname),
                                &(material.diffuse_texopt), token);

      // Set a decent diffuse default value if a diffuse texture is specified
      // without a matching Kd value.
      if (!has_kd) {
        material.diffuse[0] = static_cast<real_t>(0.6);
        material.diffuse[1] = static_cast<real_t>(0.6);
        material.diffuse[2] = static_cast<real_t>(0.6);
      }

      continue;
    }

    // specular texture
    if ((0 == strncmp(token, "map_Ks", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.specular_texname),
                                &(material.specular_texopt), token);
      continue;
    }

    // specular highlight texture
    if ((0 == strncmp(token, "map_Ns", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.specular_highlight_texname),
                                &(material.specular_highlight_texopt), token);
      continue;
    }

    // bump texture
    if ((0 == strncmp(token, "map_bump", 8)) && IS_SPACE(token[8])) {
      token += 9;
      ParseTextureNameAndOption(&(material.bump_texname),
                                &(material.bump_texopt), token);
      continue;
    }

    // bump texture
    if ((0 == strncmp(token, "map_Bump", 8)) && IS_SPACE(token[8])) {
      token += 9;
      ParseTextureNameAndOption(&(material.bump_texname),
                                &(material.bump_texopt), token);
      continue;
    }

    // bump texture
    if ((0 == strncmp(token, "bump", 4)) && IS_SPACE(token[4])) {
      token += 5;
      ParseTextureNameAndOption(&(material.bump_texname),
                                &(material.bump_texopt), token);
      continue;
    }

    // alpha texture
    if ((0 == strncmp(token, "map_d", 5)) && IS_SPACE(token[5])) {
      token += 6;
      material.alpha_texname = token;
      ParseTextureNameAndOption(&(material.alpha_texname),
                                &(material.alpha_texopt), token);
      continue;
    }

    // displacement texture
    if ((0 == strncmp(token, "disp", 4)) && IS_SPACE(token[4])) {
      token += 5;
      ParseTextureNameAndOption(&(material.displacement_texname),
                                &(material.displacement_texopt), token);
      continue;
    }

    // reflection map
    if ((0 == strncmp(token, "refl", 4)) && IS_SPACE(token[4])) {
      token += 5;
      ParseTextureNameAndOption(&(material.reflection_texname),
                                &(material.reflection_texopt), token);
      continue;
    }

    // PBR: roughness texture
    if ((0 == strncmp(token, "map_Pr", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.roughness_texname),
                                &(material.roughness_texopt), token);
      continue;
    }

    // PBR: metallic texture
    if ((0 == strncmp(token, "map_Pm", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.metallic_texname),
                                &(material.metallic_texopt), token);
      continue;
    }

    // PBR: sheen texture
    if ((0 == strncmp(token, "map_Ps", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.sheen_texname),
                                &(material.sheen_texopt), token);
      continue;
    }

    // PBR: emissive texture
    if ((0 == strncmp(token, "map_Ke", 6)) && IS_SPACE(token[6])) {
      token += 7;
      ParseTextureNameAndOption(&(material.emissive_texname),
                                &(material.emissive_texopt), token);
      continue;
    }

    // PBR: normal map texture
    if ((0 == strncmp(token, "norm", 4)) && IS_SPACE(token[4])) {
      token += 5;
      ParseTextureNameAndOption(&(material.normal_texname),
                                &(material.normal_texopt), token);
      continue;
    }

    // unknown parameter
    const char *_space = strchr(token, ' ');
    if (!_space) {
      _space = strchr(token, '\t');
    }
    if (_space) {
      std::ptrdiff_t len = _space - token;
      std::string key(token, static_cast<size_t>(len));
      std::string value = _space + 1;
      material.unknown_parameter.insert(
          std::pair<std::string, std::string>(key, value));
    }
  }
  // flush last material.
  material_map->insert(std::pair<std::string, int>(
      material.name, static_cast<int>(materials->size())));
  materials->push_back(material);

  if (warning) {
    (*warning) = warn_ss.str();
  }
}

bool MaterialFileReader::operator()(const std::string &matId,
                                    std::vector<material_t> *materials,
                                    std::map<std::string, int> *matMap,
                                    std::string *warn, std::string *err) {
  if (!m_mtlBaseDir.empty()) {
#ifdef _WIN32
    char sep = ';';
#else
    char sep = ':';
#endif

    // https://stackoverflow.com/questions/5167625/splitting-a-c-stdstring-using-tokens-e-g
    std::vector<std::string> paths;
    std::istringstream f(m_mtlBaseDir);

    std::string s;
    while (getline(f, s, sep)) {
      paths.push_back(s);
    }

    for (size_t i = 0; i < paths.size(); i++) {
      std::string filepath = JoinPath(paths[i], matId);

      std::ifstream matIStream(filepath.c_str());
      if (matIStream) {
        LoadMtl(matMap, materials, &matIStream, warn, err);

        return true;
      }
    }

    std::stringstream ss;
    ss << "Material file [ " << matId
       << " ] not found in a path : " << m_mtlBaseDir << "\n";
    if (warn) {
      (*warn) += ss.str();
    }
    return false;

  } else {
    std::string filepath = matId;
    std::ifstream matIStream(filepath.c_str());
    if (matIStream) {
      LoadMtl(matMap, materials, &matIStream, warn, err);

      return true;
    }

    std::stringstream ss;
    ss << "Material file [ " << filepath
       << " ] not found in a path : " << m_mtlBaseDir << "\n";
    if (warn) {
      (*warn) += ss.str();
    }

    return false;
  }
}

bool MaterialStreamReader::operator()(const std::string &matId,
                                      std::vector<material_t> *materials,
                                      std::map<std::string, int> *matMap,
                                      std::string *warn, std::string *err) {
  (void)err;
  (void)matId;
  if (!m_inStream) {
    std::stringstream ss;
    ss << "Material stream in error state. \n";
    if (warn) {
      (*warn) += ss.str();
    }
    return false;
  }

  LoadMtl(matMap, materials, &m_inStream, warn, err);

  return true;
}

bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, const char *filename, const char *mtl_basedir,
             bool triangulate, bool default_vcols_fallback) {
  attrib->vertices.clear();
  attrib->normals.clear();
  attrib->texcoords.clear();
  attrib->colors.clear();
  shapes->clear();

  std::stringstream errss;

  std::ifstream ifs(filename);
  if (!ifs) {
    errss << "Cannot open file [" << filename << "]\n";
    if (err) {
      (*err) = errss.str();
    }
    return false;
  }

  std::string baseDir = mtl_basedir ? mtl_basedir : "";
  if (!baseDir.empty()) {
#ifndef _WIN32
    const char dirsep = '/';
#else
    const char dirsep = '\\';
#endif
    if (baseDir[baseDir.length() - 1] != dirsep) baseDir += dirsep;
  }
  MaterialFileReader matFileReader(baseDir);

  return LoadObj(attrib, shapes, materials, warn, err, &ifs, &matFileReader,
                 triangulate, default_vcols_fallback);
}

bool LoadObj(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, std::istream *inStream,
             MaterialReader *readMatFn /*= NULL*/, bool triangulate,
             bool default_vcols_fallback) {
  std::stringstream errss;

  std::vector<real_t> v;
  std::vector<real_t> vn;
  std::vector<real_t> vt;
  std::vector<real_t> vc;
  std::vector<skin_weight_t> vw;
  std::vector<tag_t> tags;
  PrimGroup prim_group;
  std::string name;

  // material
  std::set<std::string> material_filenames;
  std::map<std::string, int> material_map;
  int material = -1;

  // smoothing group id
  unsigned int current_smoothing_id =
      0;  // Initial value. 0 means no smoothing.

  int greatest_v_idx = -1;
  int greatest_vn_idx = -1;
  int greatest_vt_idx = -1;

  shape_t shape;

  bool found_all_colors = true;

  size_t line_num = 0;
  std::string linebuf;
  while (inStream->peek() != -1) {
    safeGetline(*inStream, linebuf);

    line_num++;

    // Trim newline '\r\n' or '\n'
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\n')
        linebuf.erase(linebuf.size() - 1);
    }
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\r')
        linebuf.erase(linebuf.size() - 1);
    }

    // Skip if empty line.
    if (linebuf.empty()) {
      continue;
    }

    // Skip leading space.
    const char *token = linebuf.c_str();
    token += strspn(token, " \t");

    assert(token);
    if (token[0] == '\0') continue;  // empty line

    if (token[0] == '#') continue;  // comment line

    // vertex
    if (token[0] == 'v' && IS_SPACE((token[1]))) {
      token += 2;
      real_t x, y, z;
      real_t r, g, b;

      found_all_colors &= parseVertexWithColor(&x, &y, &z, &r, &g, &b, &token);

      v.push_back(x);
      v.push_back(y);
      v.push_back(z);

      if (found_all_colors || default_vcols_fallback) {
        vc.push_back(r);
        vc.push_back(g);
        vc.push_back(b);
      }

      continue;
    }

    // normal
    if (token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))) {
      token += 3;
      real_t x, y, z;
      parseReal3(&x, &y, &z, &token);
      vn.push_back(x);
      vn.push_back(y);
      vn.push_back(z);
      continue;
    }

    // texcoord
    if (token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))) {
      token += 3;
      real_t x, y;
      parseReal2(&x, &y, &token);
      vt.push_back(x);
      vt.push_back(y);
      continue;
    }

    // skin weight. tinyobj extension
    if (token[0] == 'v' && token[1] == 'w' && IS_SPACE((token[2]))) {
      token += 3;

      // vw <vid> <joint_0> <weight_0> <joint_1> <weight_1> ...
      // example:
      // vw 0 0 0.25 1 0.25 2 0.5

      // TODO(syoyo): Add syntax check
      int vid = 0;
      vid = parseInt(&token);

      skin_weight_t sw;

      sw.vertex_id = vid;

      while (!IS_NEW_LINE(token[0])) {
        real_t j, w;
        // joint_id should not be negative, weight may be negative
        // TODO(syoyo): # of elements check
        parseReal2(&j, &w, &token, -1.0);

        if (j < static_cast<real_t>(0)) {
          if (err) {
            std::stringstream ss;
            ss << "Failed parse `vw' line. joint_id is negative. "
                  "line "
               << line_num << ".)\n";
            (*err) += ss.str();
          }
          return false;
        }

        joint_and_weight_t jw;

        jw.joint_id = int(j);
        jw.weight = w;

        sw.weightValues.push_back(jw);

        size_t n = strspn(token, " \t\r");
        token += n;
      }

      vw.push_back(sw);
    }

    // line
    if (token[0] == 'l' && IS_SPACE((token[1]))) {
      token += 2;

      __line_t line;

      while (!IS_NEW_LINE(token[0])) {
        vertex_index_t vi;
        if (!parseTriple(&token, static_cast<int>(v.size() / 3),
                         static_cast<int>(vn.size() / 3),
                         static_cast<int>(vt.size() / 2), &vi)) {
          if (err) {
            std::stringstream ss;
            ss << "Failed parse `l' line(e.g. zero value for vertex index. "
                  "line "
               << line_num << ".)\n";
            (*err) += ss.str();
          }
          return false;
        }

        line.vertex_indices.push_back(vi);

        size_t n = strspn(token, " \t\r");
        token += n;
      }

      prim_group.lineGroup.push_back(line);

      continue;
    }

    // points
    if (token[0] == 'p' && IS_SPACE((token[1]))) {
      token += 2;

      __points_t pts;

      while (!IS_NEW_LINE(token[0])) {
        vertex_index_t vi;
        if (!parseTriple(&token, static_cast<int>(v.size() / 3),
                         static_cast<int>(vn.size() / 3),
                         static_cast<int>(vt.size() / 2), &vi)) {
          if (err) {
            std::stringstream ss;
            ss << "Failed parse `p' line(e.g. zero value for vertex index. "
                  "line "
               << line_num << ".)\n";
            (*err) += ss.str();
          }
          return false;
        }

        pts.vertex_indices.push_back(vi);

        size_t n = strspn(token, " \t\r");
        token += n;
      }

      prim_group.pointsGroup.push_back(pts);

      continue;
    }

    // face
    if (token[0] == 'f' && IS_SPACE((token[1]))) {
      token += 2;
      token += strspn(token, " \t");

      face_t face;

      face.smoothing_group_id = current_smoothing_id;
      face.vertex_indices.reserve(3);

      while (!IS_NEW_LINE(token[0])) {
        vertex_index_t vi;
        if (!parseTriple(&token, static_cast<int>(v.size() / 3),
                         static_cast<int>(vn.size() / 3),
                         static_cast<int>(vt.size() / 2), &vi)) {
          if (err) {
            std::stringstream ss;
            ss << "Failed parse `f' line(e.g. zero value for face index. line "
               << line_num << ".)\n";
            (*err) += ss.str();
          }
          return false;
        }

        greatest_v_idx = greatest_v_idx > vi.v_idx ? greatest_v_idx : vi.v_idx;
        greatest_vn_idx =
            greatest_vn_idx > vi.vn_idx ? greatest_vn_idx : vi.vn_idx;
        greatest_vt_idx =
            greatest_vt_idx > vi.vt_idx ? greatest_vt_idx : vi.vt_idx;

        face.vertex_indices.push_back(vi);
        size_t n = strspn(token, " \t\r");
        token += n;
      }

      // replace with emplace_back + std::move on C++11
      prim_group.faceGroup.push_back(face);

      continue;
    }

    // use mtl
    if ((0 == strncmp(token, "usemtl", 6))) {
      token += 6;
      std::string namebuf = parseString(&token);

      int newMaterialId = -1;
      std::map<std::string, int>::const_iterator it =
          material_map.find(namebuf);
      if (it != material_map.end()) {
        newMaterialId = it->second;
      } else {
        // { error!! material not found }
        if (warn) {
          (*warn) += "material [ '" + namebuf + "' ] not found in .mtl\n";
        }
      }

      if (newMaterialId != material) {
        // Create per-face material. Thus we don't add `shape` to `shapes` at
        // this time.
        // just clear `faceGroup` after `exportGroupsToShape()` call.
        exportGroupsToShape(&shape, prim_group, tags, material, name,
                            triangulate, v, warn);
        prim_group.faceGroup.clear();
        material = newMaterialId;
      }

      continue;
    }

    // load mtl
    if ((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6]))) {
      if (readMatFn) {
        token += 7;

        std::vector<std::string> filenames;
        SplitString(std::string(token), ' ', '\\', filenames);

        if (filenames.empty()) {
          if (warn) {
            std::stringstream ss;
            ss << "Looks like empty filename for mtllib. Use default "
                  "material (line "
               << line_num << ".)\n";

            (*warn) += ss.str();
          }
        } else {
          bool found = false;
          for (size_t s = 0; s < filenames.size(); s++) {
            if (material_filenames.count(filenames[s]) > 0) {
              found = true;
              continue;
            }

            std::string warn_mtl;
            std::string err_mtl;
            bool ok = (*readMatFn)(filenames[s].c_str(), materials,
                                   &material_map, &warn_mtl, &err_mtl);
            if (warn && (!warn_mtl.empty())) {
              (*warn) += warn_mtl;
            }

            if (err && (!err_mtl.empty())) {
              (*err) += err_mtl;
            }

            if (ok) {
              found = true;
              material_filenames.insert(filenames[s]);
              break;
            }
          }

          if (!found) {
            if (warn) {
              (*warn) +=
                  "Failed to load material file(s). Use default "
                  "material.\n";
            }
          }
        }
      }

      continue;
    }

    // group name
    if (token[0] == 'g' && IS_SPACE((token[1]))) {
      // flush previous face group.
      bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                     triangulate, v, warn);
      (void)ret;  // return value not used.

      if (shape.mesh.indices.size() > 0) {
        shapes->push_back(shape);
      }

      shape = shape_t();

      // material = -1;
      prim_group.clear();

      std::vector<std::string> names;

      while (!IS_NEW_LINE(token[0])) {
        std::string str = parseString(&token);
        names.push_back(str);
        token += strspn(token, " \t\r");  // skip tag
      }

      // names[0] must be 'g'

      if (names.size() < 2) {
        // 'g' with empty names
        if (warn) {
          std::stringstream ss;
          ss << "Empty group name. line: " << line_num << "\n";
          (*warn) += ss.str();
          name = "";
        }
      } else {
        std::stringstream ss;
        ss << names[1];

        // tinyobjloader does not support multiple groups for a primitive.
        // Currently we concatinate multiple group names with a space to get
        // single group name.

        for (size_t i = 2; i < names.size(); i++) {
          ss << " " << names[i];
        }

        name = ss.str();
      }

      continue;
    }

    // object name
    if (token[0] == 'o' && IS_SPACE((token[1]))) {
      // flush previous face group.
      bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                     triangulate, v, warn);
      (void)ret;  // return value not used.

      if (shape.mesh.indices.size() > 0 || shape.lines.indices.size() > 0 ||
          shape.points.indices.size() > 0) {
        shapes->push_back(shape);
      }

      // material = -1;
      prim_group.clear();
      shape = shape_t();

      // @todo { multiple object name? }
      token += 2;
      std::stringstream ss;
      ss << token;
      name = ss.str();

      continue;
    }

    if (token[0] == 't' && IS_SPACE(token[1])) {
      const int max_tag_nums = 8192;  // FIXME(syoyo): Parameterize.
      tag_t tag;

      token += 2;

      tag.name = parseString(&token);

      tag_sizes ts = parseTagTriple(&token);

      if (ts.num_ints < 0) {
        ts.num_ints = 0;
      }
      if (ts.num_ints > max_tag_nums) {
        ts.num_ints = max_tag_nums;
      }

      if (ts.num_reals < 0) {
        ts.num_reals = 0;
      }
      if (ts.num_reals > max_tag_nums) {
        ts.num_reals = max_tag_nums;
      }

      if (ts.num_strings < 0) {
        ts.num_strings = 0;
      }
      if (ts.num_strings > max_tag_nums) {
        ts.num_strings = max_tag_nums;
      }

      tag.intValues.resize(static_cast<size_t>(ts.num_ints));

      for (size_t i = 0; i < static_cast<size_t>(ts.num_ints); ++i) {
        tag.intValues[i] = parseInt(&token);
      }

      tag.floatValues.resize(static_cast<size_t>(ts.num_reals));
      for (size_t i = 0; i < static_cast<size_t>(ts.num_reals); ++i) {
        tag.floatValues[i] = parseReal(&token);
      }

      tag.stringValues.resize(static_cast<size_t>(ts.num_strings));
      for (size_t i = 0; i < static_cast<size_t>(ts.num_strings); ++i) {
        tag.stringValues[i] = parseString(&token);
      }

      tags.push_back(tag);

      continue;
    }

    if (token[0] == 's' && IS_SPACE(token[1])) {
      // smoothing group id
      token += 2;

      // skip space.
      token += strspn(token, " \t");  // skip space

      if (token[0] == '\0') {
        continue;
      }

      if (token[0] == '\r' || token[1] == '\n') {
        continue;
      }

      if (strlen(token) >= 3 && token[0] == 'o' && token[1] == 'f' &&
          token[2] == 'f') {
        current_smoothing_id = 0;
      } else {
        // assume number
        int smGroupId = parseInt(&token);
        if (smGroupId < 0) {
          // parse error. force set to 0.
          // FIXME(syoyo): Report warning.
          current_smoothing_id = 0;
        } else {
          current_smoothing_id = static_cast<unsigned int>(smGroupId);
        }
      }

      continue;
    }  // smoothing group id

    // Ignore unknown command.
  }

  // not all vertices have colors, no default colors desired? -> clear colors
  if (!found_all_colors && !default_vcols_fallback) {
    vc.clear();
  }

  if (greatest_v_idx >= static_cast<int>(v.size() / 3)) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }
  if (greatest_vn_idx >= static_cast<int>(vn.size() / 3)) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex normal indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }
  if (greatest_vt_idx >= static_cast<int>(vt.size() / 2)) {
    if (warn) {
      std::stringstream ss;
      ss << "Vertex texcoord indices out of bounds (line " << line_num << ".)\n\n";
      (*warn) += ss.str();
    }
  }

  bool ret = exportGroupsToShape(&shape, prim_group, tags, material, name,
                                 triangulate, v, warn);
  // exportGroupsToShape return false when `usemtl` is called in the last
  // line.
  // we also add `shape` to `shapes` when `shape.mesh` has already some
  // faces(indices)
  if (ret || shape.mesh.indices
                 .size()) {  // FIXME(syoyo): Support other prims(e.g. lines)
    shapes->push_back(shape);
  }
  prim_group.clear();  // for safety

  if (err) {
    (*err) += errss.str();
  }

  attrib->vertices.swap(v);
  attrib->vertex_weights.swap(v);
  attrib->normals.swap(vn);
  attrib->texcoords.swap(vt);
  attrib->texcoord_ws.swap(vt);
  attrib->colors.swap(vc);
  attrib->skin_weights.swap(vw);

  return true;
}

bool LoadObjWithCallback(std::istream &inStream, const callback_t &callback,
                         void *user_data /*= NULL*/,
                         MaterialReader *readMatFn /*= NULL*/,
                         std::string *warn, /* = NULL*/
                         std::string *err /*= NULL*/) {
  std::stringstream errss;

  // material
  std::set<std::string> material_filenames;
  std::map<std::string, int> material_map;
  int material_id = -1;  // -1 = invalid

  std::vector<index_t> indices;
  std::vector<material_t> materials;
  std::vector<std::string> names;
  names.reserve(2);
  std::vector<const char *> names_out;

  std::string linebuf;
  while (inStream.peek() != -1) {
    safeGetline(inStream, linebuf);

    // Trim newline '\r\n' or '\n'
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\n')
        linebuf.erase(linebuf.size() - 1);
    }
    if (linebuf.size() > 0) {
      if (linebuf[linebuf.size() - 1] == '\r')
        linebuf.erase(linebuf.size() - 1);
    }

    // Skip if empty line.
    if (linebuf.empty()) {
      continue;
    }

    // Skip leading space.
    const char *token = linebuf.c_str();
    token += strspn(token, " \t");

    assert(token);
    if (token[0] == '\0') continue;  // empty line

    if (token[0] == '#') continue;  // comment line

    // vertex
    if (token[0] == 'v' && IS_SPACE((token[1]))) {
      token += 2;
      // TODO(syoyo): Support parsing vertex color extension.
      real_t x, y, z, w;  // w is optional. default = 1.0
      parseV(&x, &y, &z, &w, &token);
      if (callback.vertex_cb) {
        callback.vertex_cb(user_data, x, y, z, w);
      }
      continue;
    }

    // normal
    if (token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))) {
      token += 3;
      real_t x, y, z;
      parseReal3(&x, &y, &z, &token);
      if (callback.normal_cb) {
        callback.normal_cb(user_data, x, y, z);
      }
      continue;
    }

    // texcoord
    if (token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))) {
      token += 3;
      real_t x, y, z;  // y and z are optional. default = 0.0
      parseReal3(&x, &y, &z, &token);
      if (callback.texcoord_cb) {
        callback.texcoord_cb(user_data, x, y, z);
      }
      continue;
    }

    // face
    if (token[0] == 'f' && IS_SPACE((token[1]))) {
      token += 2;
      token += strspn(token, " \t");

      indices.clear();
      while (!IS_NEW_LINE(token[0])) {
        vertex_index_t vi = parseRawTriple(&token);

        index_t idx;
        idx.vertex_index = vi.v_idx;
        idx.normal_index = vi.vn_idx;
        idx.texcoord_index = vi.vt_idx;

        indices.push_back(idx);
        size_t n = strspn(token, " \t\r");
        token += n;
      }

      if (callback.index_cb && indices.size() > 0) {
        callback.index_cb(user_data, &indices.at(0),
                          static_cast<int>(indices.size()));
      }

      continue;
    }

    // use mtl
    if ((0 == strncmp(token, "usemtl", 6)) && IS_SPACE((token[6]))) {
      token += 7;
      std::stringstream ss;
      ss << token;
      std::string namebuf = ss.str();

      int newMaterialId = -1;
      std::map<std::string, int>::const_iterator it =
          material_map.find(namebuf);
      if (it != material_map.end()) {
        newMaterialId = it->second;
      } else {
        // { warn!! material not found }
        if (warn && (!callback.usemtl_cb)) {
          (*warn) += "material [ " + namebuf + " ] not found in .mtl\n";
        }
      }

      if (newMaterialId != material_id) {
        material_id = newMaterialId;
      }

      if (callback.usemtl_cb) {
        callback.usemtl_cb(user_data, namebuf.c_str(), material_id);
      }

      continue;
    }

    // load mtl
    if ((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6]))) {
      if (readMatFn) {
        token += 7;

        std::vector<std::string> filenames;
        SplitString(std::string(token), ' ', '\\', filenames);

        if (filenames.empty()) {
          if (warn) {
            (*warn) +=
                "Looks like empty filename for mtllib. Use default "
                "material. \n";
          }
        } else {
          bool found = false;
          for (size_t s = 0; s < filenames.size(); s++) {
            if (material_filenames.count(filenames[s]) > 0) {
              found = true;
              continue;
            }

            std::string warn_mtl;
            std::string err_mtl;
            bool ok = (*readMatFn)(filenames[s].c_str(), &materials,
                                   &material_map, &warn_mtl, &err_mtl);

            if (warn && (!warn_mtl.empty())) {
              (*warn) += warn_mtl;  // This should be warn message.
            }

            if (err && (!err_mtl.empty())) {
              (*err) += err_mtl;
            }

            if (ok) {
              found = true;
              material_filenames.insert(filenames[s]);
              break;
            }
          }

          if (!found) {
            if (warn) {
              (*warn) +=
                  "Failed to load material file(s). Use default "
                  "material.\n";
            }
          } else {
            if (callback.mtllib_cb) {
              callback.mtllib_cb(user_data, &materials.at(0),
                                 static_cast<int>(materials.size()));
            }
          }
        }
      }

      continue;
    }

    // group name
    if (token[0] == 'g' && IS_SPACE((token[1]))) {
      names.clear();

      while (!IS_NEW_LINE(token[0])) {
        std::string str = parseString(&token);
        names.push_back(str);
        token += strspn(token, " \t\r");  // skip tag
      }

      assert(names.size() > 0);

      if (callback.group_cb) {
        if (names.size() > 1) {
          // create const char* array.
          names_out.resize(names.size() - 1);
          for (size_t j = 0; j < names_out.size(); j++) {
            names_out[j] = names[j + 1].c_str();
          }
          callback.group_cb(user_data, &names_out.at(0),
                            static_cast<int>(names_out.size()));

        } else {
          callback.group_cb(user_data, NULL, 0);
        }
      }

      continue;
    }

    // object name
    if (token[0] == 'o' && IS_SPACE((token[1]))) {
      // @todo { multiple object name? }
      token += 2;

      std::stringstream ss;
      ss << token;
      std::string object_name = ss.str();

      if (callback.object_cb) {
        callback.object_cb(user_data, object_name.c_str());
      }

      continue;
    }

#if 0  // @todo
    if (token[0] == 't' && IS_SPACE(token[1])) {
      tag_t tag;

      token += 2;
      std::stringstream ss;
      ss << token;
      tag.name = ss.str();

      token += tag.name.size() + 1;

      tag_sizes ts = parseTagTriple(&token);

      tag.intValues.resize(static_cast<size_t>(ts.num_ints));

      for (size_t i = 0; i < static_cast<size_t>(ts.num_ints); ++i) {
        tag.intValues[i] = atoi(token);
        token += strcspn(token, "/ \t\r") + 1;
      }

      tag.floatValues.resize(static_cast<size_t>(ts.num_reals));
      for (size_t i = 0; i < static_cast<size_t>(ts.num_reals); ++i) {
        tag.floatValues[i] = parseReal(&token);
        token += strcspn(token, "/ \t\r") + 1;
      }

      tag.stringValues.resize(static_cast<size_t>(ts.num_strings));
      for (size_t i = 0; i < static_cast<size_t>(ts.num_strings); ++i) {
        std::stringstream ss;
        ss << token;
        tag.stringValues[i] = ss.str();
        token += tag.stringValues[i].size() + 1;
      }

      tags.push_back(tag);
    }
#endif

    // Ignore unknown command.
  }

  if (err) {
    (*err) += errss.str();
  }

  return true;
}

bool ObjReader::ParseFromFile(const std::string &filename,
                              const ObjReaderConfig &config) {
  std::string mtl_search_path;

  if (config.mtl_search_path.empty()) {
    //
    // split at last '/'(for unixish system) or '\\'(for windows) to get
    // the base directory of .obj file
    //
    size_t pos = filename.find_last_of("/\\");
    if (pos != std::string::npos) {
      mtl_search_path = filename.substr(0, pos);
    }
  } else {
    mtl_search_path = config.mtl_search_path;
  }

  valid_ = LoadObj(&attrib_, &shapes_, &materials_, &warning_, &error_,
                   filename.c_str(), mtl_search_path.c_str(),
                   config.triangulate, config.vertex_color);

  return valid_;
}

bool ObjReader::ParseFromString(const std::string &obj_text,
                                const std::string &mtl_text,
                                const ObjReaderConfig &config) {
  std::stringbuf obj_buf(obj_text);
  std::stringbuf mtl_buf(mtl_text);

  std::istream obj_ifs(&obj_buf);
  std::istream mtl_ifs(&mtl_buf);

  MaterialStreamReader mtl_ss(mtl_ifs);

  valid_ = LoadObj(&attrib_, &shapes_, &materials_, &warning_, &error_,
                   &obj_ifs, &mtl_ss, config.triangulate, config.vertex_color);

  return valid_;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
}  // namespace tinyobj

#endif

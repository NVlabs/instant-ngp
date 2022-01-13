// Use double precision for better python integration.
// Need also define this in `binding.cc`(and all compilation units)
#define TINYOBJLOADER_USE_DOUBLE

// Use robust triangulation by using Mapbox earcut.
#define TINYOBJLOADER_USE_MAPBOX_EARCUT

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file   NanoVDB.h

    \author Ken Museth

    \date  January 8, 2020

    \brief Implements a light-weight self-contained VDB data-structure in a
           single file! In other words, this is a significantly watered-down
           version of the OpenVDB implementation, with few dependencies - so
           a one-stop-shop for a minimalistic VDB data structure that run on
           most platforms!

    \note It is important to note that NanoVDB (by design) is a read-only
          sparse GPU (and CPU) friendly data structure intended for applications
          like rendering and collision detection. As such it obviously lacks
          a lot of the functionality and features of OpenVDB grids. NanoVDB
          is essentially a compact linearized (or serialized) representation of
          an OpenVDB tree with getValue methods only. For best performance use
          the ReadAccessor::getValue method as opposed to the Tree::getValue
          method. Note that since a ReadAccessor caches previous access patterns
          it is by design not thread-safe, so use one instantiation per thread
          (it is very light-weight). Also, it is not safe to copy accessors between
          the GPU and CPU! In fact, client code should only interface
          with the API of the Grid class (all other nodes of the NanoVDB data
          structure can safely be ignored by most client codes)!


    \warning NanoVDB grids can only be constructed via tools like openToNanoVDB
             or the GridBuilder. This explains why none of the grid nodes defined below
             have public constructors or destructors.

    \details Please see the following paper for more details on the data structure:
          K. Museth, “VDB: High-Resolution Sparse Volumes with Dynamic Topology”,
          ACM Transactions on Graphics 32(3), 2013, which can be found here:
          http://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf


    Overview: This file implements the following fundamental class that when combined
          forms the backbone of the VDB tree data structure:

          Coord- a signed integer coordinate
          Vec3 - a 3D vector
          Vec4 - a 4D vector
          BBox - a bounding box
          Mask - a bitmask essential to the non-root tree nodes
          Map  - an affine coordinate transformation
          Grid - contains a Tree and a map for world<->index transformations. Use
                 this class as the main API with client code!
          Tree - contains a RootNode and getValue methods that should only be used for debugging
          RootNode - the top-level node of the VDB data structure
          InternalNode - the internal nodes of the VDB data structure
          LeafNode - the lowest level tree nodes that encode voxel values and state
          ReadAccessor - implements accelerated random access operations

    Semantics: A VDB data structure encodes values and (binary) states associated with
          signed integer coordinates. Values encoded at the leaf node level are
          denoted voxel values, and values associated with other tree nodes are referred
          to as tile values, which by design cover a larger coordinate index domain.


    Memory layout:

    GridData is always at the very beginning of the buffer immediately followed by TreeData!
    The remaining nodes and blind-data are allowed to be scattered thoughout the buffer,
    though in practice they are arranged as:

    GridData: 672 bytes (e.g. magic, checksum, major, flags, index, count, size, name, map, world bbox, voxel size, class, type, offset, count)

    TreeData: 64 bytes (node counts and byte offsets)

    ... optional padding ...

    RootData: size depends on ValueType (index bbox, voxel count, tile count, min/max/avg/standard deviation)

    Array of: RootData::Tile

    ... optional padding ...

    Array of: Upper InternalNodes of size 32^3:  bbox, two bit masks, 32768 tile values, and min/max/avg/standard deviation values

    ... optional padding ...

    Array of: Lower InternalNodes of size 16^3:  bbox, two bit masks, 4096 tile values, and min/max/avg/standard deviation values

    ... optional padding ...

    Array of: LeafNodes of size 8^3: bbox, bit masks, 512 voxel values, and min/max/avg/standard deviation values


    Example layout: ("---" implies it has a custom offset, "..." implies zero or more)
    [GridData(672B)][TreeData(64B)]---[RootData][N x Root::Tile]---[NodeData<5>]---[ModeData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.

*/

#ifndef NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED
#define NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED

#define NANOVDB_MAGIC_NUMBER 0x304244566f6e614eUL // "NanoVDB0" in hex - little endian (uint64_t)

#define NANOVDB_MAJOR_VERSION_NUMBER 32 // reflects changes to the ABI and hence also the file format
#define NANOVDB_MINOR_VERSION_NUMBER 3 //  reflects changes to the API but not ABI
#define NANOVDB_PATCH_VERSION_NUMBER 3 //  reflects changes that does not affect the ABI or API

// This replaces a Coord key at the root level with a single uint64_t
#define USE_SINGLE_ROOT_KEY

// This replaces three levels of Coord keys in the ReadAccessor with one Coord
//#define USE_SINGLE_ACCESSOR_KEY

#define NANOVDB_FPN_BRANCHLESS

#define NANOVDB_DATA_ALIGNMENT 32

#if !defined(NANOVDB_ALIGN)
#define NANOVDB_ALIGN(n) alignas(n)
#endif // !defined(NANOVDB_ALIGN)

#ifdef __CUDACC_RTC__

typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned int       uint32_t;
typedef unsigned short     uint16_t;
typedef unsigned long long uint64_t;

#define NANOVDB_ASSERT(x)

#define UINT64_C(x)  (x ## ULL)

#else // __CUDACC_RTC__

#include <stdlib.h> //    for abs in clang7
#include <stdint.h> //    for types like int32_t etc
#include <stddef.h> //    for size_t type
#include <cassert> //     for assert
#include <cstdio> //      for sprinf
#include <cmath> //       for sqrt and fma
#include <limits> //      for numeric_limits

// All asserts can be disabled here, even for debug builds
#if 1
#define NANOVDB_ASSERT(x) assert(x)
#else
#define NANOVDB_ASSERT(x)
#endif

#if defined(NANOVDB_USE_INTRINSICS) && defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse64)
#pragma intrinsic(_BitScanForward64)
#endif

#endif // __CUDACC_RTC__

#if defined(__CUDACC__) || defined(__HIP__)
// Only define __hostdev__ when using NVIDIA CUDA or HIP compiler
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif

// The following macro will suppress annoying warnings when nvcc
// compiles functions that call (host) intrinsics (which is perfectly valid)
#if defined(_MSC_VER) && defined(__CUDACC__)
#define NANOVDB_HOSTDEV_DISABLE_WARNING __pragma("hd_warning_disable")
#elif defined(__GNUC__) && defined(__CUDACC__)
#define NANOVDB_HOSTDEV_DISABLE_WARNING _Pragma("hd_warning_disable")
#else
#define NANOVDB_HOSTDEV_DISABLE_WARNING
#endif

// A portable implementation of offsetof - unfortunately it doesn't work with static_assert
#define NANOVDB_OFFSETOF(CLASS, MEMBER) ((int)(size_t)((char*)&((CLASS*)0)->MEMBER - (char*)0))

namespace nanovdb {

// --------------------------> Build types <------------------------------------

/// @brief Dummy type for a voxel with a binary mask value, e.g. the active state
class ValueMask {};

/// @brief Dummy type for a 16 bit floating point values
class Half {};

/// @brief Dummy type for a 4bit quantization of float point values
class Fp4 {};

/// @brief Dummy type for a 8bit quantization of float point values
class Fp8 {};

/// @brief Dummy type for a 16bit quantization of float point values
class Fp16 {};

/// @brief Dummy type for a variable bit quantization of floating point values
class FpN {};

// --------------------------> GridType <------------------------------------

/// @brief List of types that are currently supported by NanoVDB
///
/// @note To expand on this list do:
///       1) Add the new type between Unknown and End in the enum below
///       2) Add the new type to OpenToNanoVDB::processGrid that maps OpenVDB types to GridType
///       3) Verify that the ConvertTrait in NanoToOpenVDB.h works correctly with the new type
///       4) Add the new type to mapToGridType (defined below) that maps NanoVDB types to GridType
///       5) Add the new type to toStr (defined below)
enum class GridType : uint32_t { Unknown = 0,
                                 Float   = 1, //  single precision floating point value
                                 Double  = 2,//   double precision floating point value
                                 Int16   = 3,//   half precision signed integer value
                                 Int32   = 4,//   single precision signed integer value
                                 Int64   = 5,//   double precision signed integer value
                                 Vec3f   = 6,//   single precision floating 3D vector
                                 Vec3d   = 7,//   double precision floating 3D vector
                                 Mask    = 8,//   no value, just the active state
                                 Half    = 9,//   half precision floating point value
                                 UInt32  = 10,// single precision unsigned integer value
                                 Boolean = 11,// boolean value, encoded in bit array
                                 RGBA8   = 12,// RGBA packed into 32bit word in reverse-order. R in low bits.
                                 Fp4     = 13,// 4bit quantization of float point value
                                 Fp8     = 14,// 8bit quantization of float point value
                                 Fp16    = 15,// 16bit quantization of float point value
                                 FpN     = 16,// variable bit quantization of floating point value
                                 Vec4f   = 17,// single precision floating 4D vector
                                 Vec4d   = 18,// double precision floating 4D vector
                                 End     = 19 };

#ifndef __CUDACC_RTC__
/// @brief Retuns a c-string used to describe a GridType
inline const char* toStr(GridType gridType)
{
    static const char * LUT[] = { "?", "float", "double" , "int16", "int32",
                                 "int64", "Vec3f", "Vec3d", "Mask", "Half",
                                 "uint32", "bool", "RGBA8", "Float4", "Float8",
                                 "Float16", "FloatN", "Vec4f", "Vec4d", "End" };
    static_assert( sizeof(LUT)/sizeof(char*) - 1 == int(GridType::End), "Unexpected size of LUT" );
    return LUT[static_cast<int>(gridType)];
}
#endif

// --------------------------> GridClass <------------------------------------

/// @brief Classes (defined in OpenVDB) that are currently supported by NanoVDB
enum class GridClass : uint32_t { Unknown = 0,
                                  LevelSet = 1, //   narrow band level set, e.g. SDF
                                  FogVolume = 2, //  fog volume, e.g. density
                                  Staggered = 3, //  staggered MAC grid, e.g. velocity
                                  PointIndex = 4, // point index grid
                                  PointData = 5, //  point data grid
                                  Topology = 6, // grid with active states only (no values)
                                  VoxelVolume = 7, // volume of geometric cubes, e.g. minecraft
                                  End = 8 };

#ifndef __CUDACC_RTC__
/// @brief Retuns a c-string used to describe a GridClass
inline const char* toStr(GridClass gridClass)
{
    static const char * LUT[] = { "?", "SDF", "FOG" , "MAC", "PNTIDX",
                                 "PNTDAT", "TOPO", "VOX", "END" };
    static_assert( sizeof(LUT)/sizeof(char*) - 1 == int(GridClass::End), "Unexpected size of LUT" );
    return LUT[static_cast<int>(gridClass)];
}
#endif

// --------------------------> GridFlags <------------------------------------

/// @brief Grid flags which indicate what extra information is present in the grid buffer.
enum class GridFlags : uint32_t {
    HasLongGridName = 1 << 0,// grid name is longer than 256 characters
    HasBBox = 1 << 1,// nodes contain bounding-boxes of active values
    HasMinMax = 1 << 2,// nodes contain min/max of active values
    HasAverage = 1 << 3,// nodes contain averages of active values
    HasStdDeviation = 1 << 4,// nodes contain standard deviations of active values
    IsBreadthFirst = 1 << 5,// nodes are arranged breadth-first in memory
    End = 1 << 6,
};

#ifndef __CUDACC_RTC__
/// @brief Retuns a c-string used to describe a GridFlags
inline const char* toStr(GridFlags gridFlags)
{
    static const char * LUT[] = { "has long grid name",
                                  "has bbox",
                                  "has min/max",
                                  "has average",
                                  "has standard deviation",
                                  "is breadth-first",
                                  "end" };
    static_assert( 1 << (sizeof(LUT)/sizeof(char*) - 1) == int(GridFlags::End), "Unexpected size of LUT" );
    return LUT[static_cast<int>(gridFlags)];
}
#endif

// --------------------------> GridBlindData enums <------------------------------------

/// @brief Blind-data Classes that are currently supported by NanoVDB
enum class GridBlindDataClass : uint32_t { Unknown = 0,
                                           IndexArray = 1,
                                           AttributeArray = 2,
                                           GridName = 3,
                                           End = 4 };

/// @brief Blind-data Semantics that are currently understood by NanoVDB
enum class GridBlindDataSemantic : uint32_t { Unknown = 0,
                                              PointPosition = 1,
                                              PointColor = 2,
                                              PointNormal = 3,
                                              PointRadius = 4,
                                              PointVelocity = 5,
                                              PointId = 6,
                                              End = 7 };

// --------------------------> is_same <------------------------------------

/// @brief C++11 implementation of std::is_same
template<typename T1, typename T2>
struct is_same
{
    static constexpr bool value = false;
};

template<typename T>
struct is_same<T, T>
{
    static constexpr bool value = true;
};

// --------------------------> enable_if <------------------------------------

/// @brief C++11 implementation of std::enable_if
template <bool, typename T = void>
struct enable_if
{
};

template <typename T>
struct enable_if<true, T>
{
    using type = T;
};

// --------------------------> is_floating_point <------------------------------------

/// @brief C++11 implementation of std::is_floating_point
template<typename T>
struct is_floating_point
{
    static const bool value = is_same<T, float>::value || is_same<T, double>::value;
};

// --------------------------> is_specialization <------------------------------------

/// @brief Metafunction used to determine if the first template
///        parameter is a specialization of the class template
///        given in the second template parameter.
///
/// @details is_specialization<Vec3<float>, Vec3>::value == true;
template<typename AnyType, template<typename...> class TemplateType>
struct is_specialization
{
    static const bool value = false;
};
template<typename... Args, template<typename...> class TemplateType>
struct is_specialization<TemplateType<Args...>, TemplateType>
{
    static const bool value = true;
};

// --------------------------> Value Map <------------------------------------

/// @brief Maps one type (e.g. the build types above) to other (actual) types
template <typename T>
struct BuildToValueMap
{
    using Type = T;
    using type = T;
};

template<>
struct BuildToValueMap<ValueMask>
{
    using Type = bool;
    using type = bool;
};

template<>
struct BuildToValueMap<Half>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp4>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp8>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<Fp16>
{
    using Type = float;
    using type = float;
};

template<>
struct BuildToValueMap<FpN>
{
    using Type = float;
    using type = float;
};

// --------------------------> PtrDiff  PtrAdd <------------------------------------

template <typename T1, typename T2>
__hostdev__ inline static int64_t PtrDiff(const T1* p, const T2* q)
{
    NANOVDB_ASSERT(p && q);
    return reinterpret_cast<const char*>(p) - reinterpret_cast<const char*>(q);
}

template <typename DstT, typename SrcT>
__hostdev__ inline static DstT* PtrAdd(SrcT *p, int64_t offset)
{
    NANOVDB_ASSERT(p);
    return reinterpret_cast<DstT*>(reinterpret_cast<char*>(p) + offset);
}

template <typename DstT, typename SrcT>
__hostdev__ inline static const DstT* PtrAdd(const SrcT *p, int64_t offset)
{
    NANOVDB_ASSERT(p);
    return reinterpret_cast<const DstT*>(reinterpret_cast<const char*>(p) + offset);
}
// --------------------------> Rgba8 <------------------------------------

/// @brief 8-bit red, green, blue, alpha packed into 32 bit unsigned int
class Rgba8
{
    union {
        uint8_t  c[4];// 4 color channels of red, green, blue and alpha components.
        uint32_t packed;// 32 bit packed representation
    } mData;
public:
    static const int SIZE = 4;
    using ValueType = uint8_t;

    Rgba8(const Rgba8&) = default;
    Rgba8(Rgba8&&) = default;
    Rgba8& operator=(Rgba8&&) = default;
    Rgba8& operator=(const Rgba8&) = default;
    __hostdev__ Rgba8() : mData{0,0,0,0} {static_assert(sizeof(uint32_t) == sizeof(Rgba8),"Unexpected sizeof");}
    __hostdev__ Rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255u) : mData{r, g, b, a} {}
    explicit __hostdev__ Rgba8(uint8_t v) : Rgba8(v,v,v,v) {}
    __hostdev__ Rgba8(float r, float g, float b, float a = 1.0f)
        : mData{(uint8_t(0.5f + r * 255.0f)),// round to nearest
                (uint8_t(0.5f + g * 255.0f)),// round to nearest
                (uint8_t(0.5f + b * 255.0f)),// round to nearest
                (uint8_t(0.5f + a * 255.0f))}// round to nearest
    {
    }
    __hostdev__ bool operator<(const Rgba8& rhs) const { return mData.packed < rhs.mData.packed; }
    __hostdev__ bool operator==(const Rgba8& rhs) const { return mData.packed == rhs.mData.packed; }
    __hostdev__ float lengthSqr() const
    {
        return 0.0000153787005f*(float(mData.c[0])*mData.c[0] +
                                 float(mData.c[1])*mData.c[1] +
                                 float(mData.c[2])*mData.c[2]);//1/255^2
    }
    __hostdev__ float length() const { return sqrtf(this->lengthSqr() ); }
    __hostdev__ const uint8_t& operator[](int n) const { return mData.c[n]; }
    __hostdev__ uint8_t& operator[](int n) { return mData.c[n]; }
    __hostdev__ const uint32_t& packed() const { return mData.packed; }
    __hostdev__ uint32_t& packed() { return mData.packed; }
    __hostdev__ const uint8_t& r() const { return mData.c[0]; }
    __hostdev__ const uint8_t& g() const { return mData.c[1]; }
    __hostdev__ const uint8_t& b() const { return mData.c[2]; }
    __hostdev__ const uint8_t& a() const { return mData.c[3]; }
    __hostdev__ uint8_t& r() { return mData.c[0]; }
    __hostdev__ uint8_t& g() { return mData.c[1]; }
    __hostdev__ uint8_t& b() { return mData.c[2]; }
    __hostdev__ uint8_t& a() { return mData.c[3]; }
};// Rgba8

using PackedRGBA8 = Rgba8;// for backwards compatibility

// --------------------------> isValue(GridType, GridClass) <------------------------------------

/// @brief return true if the GridType maps to a floating point value.
__hostdev__ inline bool isFloatingPoint(GridType gridType)
{
    return gridType == GridType::Float  ||
           gridType == GridType::Double ||
           gridType == GridType::Fp4    ||
           gridType == GridType::Fp8    ||
           gridType == GridType::Fp16   ||
           gridType == GridType::FpN;
}

// --------------------------> isValue(GridType, GridClass) <------------------------------------

/// @brief return true if the combination of GridType and GridClass is valid.
__hostdev__ inline bool isValid(GridType gridType, GridClass gridClass)
{
    if (gridClass == GridClass::LevelSet || gridClass == GridClass::FogVolume) {
        return isFloatingPoint(gridType);
    } else if (gridClass == GridClass::Staggered) {
        return gridType == GridType::Vec3f || gridType == GridType::Vec3d ||
               gridType == GridType::Vec4f || gridType == GridType::Vec4d;
    } else if (gridClass == GridClass::PointIndex || gridClass ==  GridClass::PointData) {
        return gridType == GridType::UInt32;
    } else if (gridClass == GridClass::VoxelVolume) {
        return gridType == GridType::RGBA8 || gridType == GridType::Float || gridType == GridType::Double || gridType == GridType::Vec3f || gridType == GridType::Vec3d || gridType == GridType::UInt32;
    }
    return gridClass < GridClass::End && gridType < GridType::End;// any valid combination
}

// ----------------------------> Version class <-------------------------------------

/// @brief Bit-compacted representation of all three version numbers
///
/// @details major is the top 11 bits, minor is the 11 middle bits and patch is the lower 10 bits
class Version
{
    uint32_t mData;// 11 + 11 + 10 bit packing of major + minor + patch
public:
    __hostdev__ Version() : mData( uint32_t(NANOVDB_MAJOR_VERSION_NUMBER) << 21 |
                                   uint32_t(NANOVDB_MINOR_VERSION_NUMBER) << 10 |
                                   uint32_t(NANOVDB_PATCH_VERSION_NUMBER) )
    {
    }
    __hostdev__ Version(uint32_t major, uint32_t minor, uint32_t patch)
        : mData( major << 21 | minor << 10 | patch )
    {
        NANOVDB_ASSERT(major < (1u << 11));// max value of major is 2047
        NANOVDB_ASSERT(minor < (1u << 11));// max value of minor is 2047
        NANOVDB_ASSERT(patch < (1u << 10));// max value of patch is 1023
    }
    __hostdev__ bool operator==(const Version &rhs) const {return mData == rhs.mData;}
    __hostdev__ bool operator< (const Version &rhs) const {return mData <  rhs.mData;}
    __hostdev__ bool operator<=(const Version &rhs) const {return mData <= rhs.mData;}
    __hostdev__ bool operator> (const Version &rhs) const {return mData >  rhs.mData;}
    __hostdev__ bool operator>=(const Version &rhs) const {return mData >= rhs.mData;}
    __hostdev__ uint32_t id()       const { return mData; }
    __hostdev__ uint32_t getMajor() const { return (mData >> 21) & ((1u << 11) - 1);}
    __hostdev__ uint32_t getMinor() const { return (mData >> 10) & ((1u << 11) - 1);}
    __hostdev__ uint32_t getPatch() const { return  mData        & ((1u << 10) - 1);}

#ifndef __CUDACC_RTC__
    const char* c_str() const
    {
        char *buffer = (char*)malloc(4 + 1 + 4 + 1 + 4 + 1);// xxxx.xxxx.xxxx\n
        sprintf(buffer, "%d.%d.%d", this->getMajor(), this->getMinor(), this->getPatch());
        return buffer;
    }
#endif
};// Version

// ----------------------------> Various math functions <-------------------------------------

//@{
/// Tolerance for floating-point comparison
template<typename T>
struct Tolerance;
template<>
struct Tolerance<float>
{
    __hostdev__ static float value() { return 1e-8f; }
};
template<>
struct Tolerance<double>
{
    __hostdev__ static double value() { return 1e-15; }
};
//@}

//@{
/// Delta for small floating-point offsets
template<typename T>
struct Delta;
template<>
struct Delta<float>
{
    __hostdev__ static float value() { return 1e-5f; }
};
template<>
struct Delta<double>
{
    __hostdev__ static double value() { return 1e-9; }
};
//@}

//@{
/// Maximum floating-point values
template<typename T>
struct Maximum;
#if defined(__CUDA_ARCH__) || defined(__HIP__)
template<>
struct Maximum<int>
{
    __hostdev__ static int value() { return 2147483647; }
};
template<>
struct Maximum<uint32_t>
{
    __hostdev__ static uint32_t value() { return 4294967295; }
};
template<>
struct Maximum<float>
{
    __hostdev__ static float value() { return 1e+38f; }
};
template<>
struct Maximum<double>
{
    __hostdev__ static double value() { return 1e+308; }
};
#else
template<typename T>
struct Maximum
{
    static T value() { return std::numeric_limits<T>::max(); }
};
#endif
//@}

template<typename Type>
__hostdev__ inline bool isApproxZero(const Type& x)
{
    return !(x > Tolerance<Type>::value()) && !(x < -Tolerance<Type>::value());
}

template<typename Type>
__hostdev__ inline Type Min(Type a, Type b)
{
    return (a < b) ? a : b;
}
__hostdev__ inline int32_t Min(int32_t a, int32_t b)
{
    return int32_t(fminf(float(a), float(b)));
}
__hostdev__ inline uint32_t Min(uint32_t a, uint32_t b)
{
    return uint32_t(fminf(float(a), float(b)));
}
__hostdev__ inline float Min(float a, float b)
{
    return fminf(a, b);
}
__hostdev__ inline double Min(double a, double b)
{
    return fmin(a, b);
}
template<typename Type>
__hostdev__ inline Type Max(Type a, Type b)
{
    return (a > b) ? a : b;
}

__hostdev__ inline int32_t Max(int32_t a, int32_t b)
{
    return int32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline uint32_t Max(uint32_t a, uint32_t b)
{
    return uint32_t(fmaxf(float(a), float(b)));
}
__hostdev__ inline float Max(float a, float b)
{
    return fmaxf(a, b);
}
__hostdev__ inline double Max(double a, double b)
{
    return fmax(a, b);
}
__hostdev__ inline float Clamp(float x, float a, float b)
{
    return Max(Min(x, b), a);
}
__hostdev__ inline double Clamp(double x, double a, double b)
{
    return Max(Min(x, b), a);
}

__hostdev__ inline float Fract(float x)
{
    return x - floorf(x);
}
__hostdev__ inline double Fract(double x)
{
    return x - floor(x);
}

__hostdev__ inline int32_t Floor(float x)
{
    return int32_t(floorf(x));
}
__hostdev__ inline int32_t Floor(double x)
{
    return int32_t(floor(x));
}

__hostdev__ inline int32_t Ceil(float x)
{
    return int32_t(ceilf(x));
}
__hostdev__ inline int32_t Ceil(double x)
{
    return int32_t(ceil(x));
}

template<typename T>
__hostdev__ inline T Pow2(T x)
{
    return x * x;
}

template<typename T>
__hostdev__ inline T Pow3(T x)
{
    return x * x * x;
}

template<typename T>
__hostdev__ inline T Pow4(T x)
{
    return Pow2(x * x);
}
template<typename T>
__hostdev__ inline T Abs(T x)
{
    return x < 0 ? -x : x;
}

template<>
__hostdev__ inline float Abs(float x)
{
    return fabs(x);
}

template<>
__hostdev__ inline double Abs(double x)
{
    return fabs(x);
}

template<>
__hostdev__ inline int Abs(int x)
{
    return abs(x);
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<RealT>& xyz);

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<float>& xyz)
{
    return CoordT(int32_t(rintf(xyz[0])), int32_t(rintf(xyz[1])), int32_t(rintf(xyz[2])));
    //return CoordT(int32_t(roundf(xyz[0])), int32_t(roundf(xyz[1])), int32_t(roundf(xyz[2])) );
    //return CoordT(int32_t(floorf(xyz[0] + 0.5f)), int32_t(floorf(xyz[1] + 0.5f)), int32_t(floorf(xyz[2] + 0.5f)));
}

template<typename CoordT, template<typename> class Vec3T>
__hostdev__ inline CoordT Round(const Vec3T<double>& xyz)
{
    return CoordT(int32_t(floor(xyz[0] + 0.5)), int32_t(floor(xyz[1] + 0.5)), int32_t(floor(xyz[2] + 0.5)));
}

template<typename CoordT, typename RealT, template<typename> class Vec3T>
__hostdev__ inline CoordT RoundDown(const Vec3T<RealT>& xyz)
{
    return CoordT(Floor(xyz[0]), Floor(xyz[1]), Floor(xyz[2]));
}

//@{
/// Return the square root of a floating-point value.
__hostdev__ inline float Sqrt(float x)
{
    return sqrtf(x);
}
__hostdev__ inline double Sqrt(double x)
{
    return sqrt(x);
}
//@}

/// Return the sign of the given value as an integer (either -1, 0 or 1).
template <typename T>
__hostdev__ inline T Sign(const T &x) { return ((T(0) < x)?T(1):T(0)) - ((x < T(0))?T(1):T(0)); }

template<typename Vec3T>
__hostdev__ inline int MinIndex(const Vec3T& v)
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] < v[1]) << 2) + ((v[0] < v[2]) << 1) + (v[1] < v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] < v[1] && v[0] < v[2])
        return 0;
    if (v[1] < v[2])
        return 1;
    else
        return 2;
#endif
}

template<typename Vec3T>
__hostdev__ inline int MaxIndex(const Vec3T& v)
{
#if 0
    static const int hashTable[8] = {2, 1, 9, 1, 2, 9, 0, 0}; //9 are dummy values
    const int        hashKey = ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]); // ?*4+?*2+?*1
    return hashTable[hashKey];
#else
    if (v[0] > v[1] && v[0] > v[2])
        return 0;
    if (v[1] > v[2])
        return 1;
    else
        return 2;
#endif
}

/// @brief round up byteSize to the nearest wordSize, e.g. to align to machine word: AlignUp<sizeof(size_t)(n)
///
/// @details both wordSize and byteSize are in byte units
template<uint64_t wordSize>
__hostdev__ inline uint64_t AlignUp(uint64_t byteCount)
{
    const uint64_t r = byteCount % wordSize;
    return r ? byteCount - r + wordSize : byteCount;
}

// ------------------------------> Coord <--------------------------------------

// forward decleration so we can define Coord::asVec3s and Coord::asVec3d
template<typename> class Vec3;

/// @brief Signed (i, j, k) 32-bit integer coordinate class, similar to openvdb::math::Coord
class Coord
{
    int32_t mVec[3]; // private member data - three signed index coordinates
public:
    using ValueType = int32_t;
    using IndexType = uint32_t;

    /// @brief Initialize all coordinates to zero.
    __hostdev__ Coord()
        : mVec{0, 0, 0}
    {
    }

    /// @brief Initializes all coordinates to the given signed integer.
    __hostdev__ explicit Coord(ValueType n)
        : mVec{n, n, n}
    {
    }

    /// @brief Initializes coordinate to the given signed integers.
    __hostdev__ Coord(ValueType i, ValueType j, ValueType k)
        : mVec{i, j, k}
    {
    }

    __hostdev__ Coord(ValueType *ptr)
        : mVec{ptr[0], ptr[1], ptr[2]}
    {
    }

    __hostdev__ int32_t x() const { return mVec[0]; }
    __hostdev__ int32_t y() const { return mVec[1]; }
    __hostdev__ int32_t z() const { return mVec[2]; }

    __hostdev__ int32_t& x() { return mVec[0]; }
    __hostdev__ int32_t& y() { return mVec[1]; }
    __hostdev__ int32_t& z() { return mVec[2]; }

    __hostdev__ static Coord max() { return Coord(int32_t((1u << 31) - 1)); }

    __hostdev__ static Coord min() { return Coord(-int32_t((1u << 31) - 1) - 1); }

    __hostdev__ static size_t memUsage() { return sizeof(Coord); }

    /// @brief Return a const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ const ValueType& operator[](IndexType i) const { return mVec[i]; }

    /// @brief Return a non-const reference to the given Coord component.
    /// @warning The argument is assumed to be 0, 1, or 2.
    __hostdev__ ValueType& operator[](IndexType i) { return mVec[i]; }

   /// @brief Assignment operator that works with openvdb::Coord
   template <typename CoordT>
    __hostdev__ Coord& operator=(const CoordT &other)
    {
        static_assert(sizeof(Coord) == sizeof(CoordT), "Mis-matched sizeof");
        mVec[0] = other[0];
        mVec[1] = other[1];
        mVec[2] = other[2];
        return *this;
    }

    /// @brief Return a new instance with coordinates masked by the given unsigned integer.
    __hostdev__ Coord operator&(IndexType n) const { return Coord(mVec[0] & n, mVec[1] & n, mVec[2] & n); }

    // @brief Return a new instance with coordinates left-shifted by the given unsigned integer.
    __hostdev__ Coord operator<<(IndexType n) const { return Coord(mVec[0] << n, mVec[1] << n, mVec[2] << n); }

    // @brief Return a new instance with coordinates right-shifted by the given unsigned integer.
    __hostdev__ Coord operator>>(IndexType n) const { return Coord(mVec[0] >> n, mVec[1] >> n, mVec[2] >> n); }

    /// @brief Return true if this Coord is lexicographically less than the given Coord.
    __hostdev__ bool operator<(const Coord& rhs) const
    {
        return mVec[0] < rhs[0] ? true : mVec[0] > rhs[0] ? false : mVec[1] < rhs[1] ? true : mVec[1] > rhs[1] ? false : mVec[2] < rhs[2] ? true : false;
    }

    // @brief Return true if the Coord components are identical.
    __hostdev__ bool operator==(const Coord& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ bool operator!=(const Coord& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    __hostdev__ Coord& operator&=(int n)
    {
        mVec[0] &= n;
        mVec[1] &= n;
        mVec[2] &= n;
        return *this;
    }
    __hostdev__ Coord& operator<<=(uint32_t n)
    {
        mVec[0] <<= n;
        mVec[1] <<= n;
        mVec[2] <<= n;
        return *this;
    }
    __hostdev__ Coord& operator+=(int n)
    {
        mVec[0] += n;
        mVec[1] += n;
        mVec[2] += n;
        return *this;
    }
    __hostdev__ Coord operator+(const Coord& rhs) const { return Coord(mVec[0] + rhs[0], mVec[1] + rhs[1], mVec[2] + rhs[2]); }
    __hostdev__ Coord operator-(const Coord& rhs) const { return Coord(mVec[0] - rhs[0], mVec[1] - rhs[1], mVec[2] - rhs[2]); }
    __hostdev__ Coord& operator+=(const Coord& rhs)
    {
        mVec[0] += rhs[0];
        mVec[1] += rhs[1];
        mVec[2] += rhs[2];
        return *this;
    }
    __hostdev__ Coord& operator-=(const Coord& rhs)
    {
        mVec[0] -= rhs[0];
        mVec[1] -= rhs[1];
        mVec[2] -= rhs[2];
        return *this;
    }

    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Coord& minComponent(const Coord& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Coord& maxComponent(const Coord& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    __hostdev__ Coord offsetBy(ValueType dx, ValueType dy, ValueType dz) const
    {
        return Coord(mVec[0] + dx, mVec[1] + dy, mVec[2] + dz);
    }

    __hostdev__ Coord offsetBy(ValueType n) const { return this->offsetBy(n, n, n); }

    /// Return true if any of the components of @a a are smaller than the
    /// corresponding components of @a b.
    __hostdev__ static inline bool lessThan(const Coord& a, const Coord& b)
    {
        return (a[0] < b[0] || a[1] < b[1] || a[2] < b[2]);
    }

    /// @brief Return the largest integer coordinates that are not greater
    /// than @a xyz (node centered conversion).
    template<typename Vec3T>
    __hostdev__ static Coord Floor(const Vec3T& xyz) { return Coord(nanovdb::Floor(xyz[0]), nanovdb::Floor(xyz[1]), nanovdb::Floor(xyz[2])); }

    /// @brief Return a hash key derived from the existing coordinates.
    /// @details For details on this hash function please see the VDB paper.
    template<int Log2N = 3 + 4 + 5>
    __hostdev__ uint32_t hash() const { return ((1 << Log2N) - 1) & (mVec[0] * 73856093 ^ mVec[1] * 19349663 ^ mVec[2] * 83492791); }

    /// @brief Return the octant of this Coord
    //__hostdev__ size_t octant() const { return (uint32_t(mVec[0])>>31) | ((uint32_t(mVec[1])>>31)<<1) | ((uint32_t(mVec[2])>>31)<<2); }
    __hostdev__ uint8_t octant() const { return (uint8_t(bool(mVec[0] & (1u << 31)))) |
                                                (uint8_t(bool(mVec[1] & (1u << 31))) << 1) |
                                                (uint8_t(bool(mVec[2] & (1u << 31))) << 2); }

    /// @brief Return a single precision floating-point vector of this coordinate
    __hostdev__ inline Vec3<float> asVec3s() const;

    /// @brief Return a double precision floating-point vector of this coordinate
    __hostdev__ inline Vec3<double> asVec3d() const;
}; // Coord class

// ----------------------------> Vec3 <--------------------------------------

/// @brief A simple vector class with three double components, similar to openvdb::math::Vec3
template<typename T>
class Vec3
{
    T mVec[3];

public:
    static const int SIZE = 3;
    using ValueType = T;
    Vec3() = default;
    __hostdev__ explicit Vec3(T x)
        : mVec{x, x, x}
    {
    }
    __hostdev__ Vec3(T x, T y, T z)
        : mVec{x, y, z}
    {
    }
    template<typename T2>
    __hostdev__ explicit Vec3(const Vec3<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2])}
    {
    }
    __hostdev__ explicit Vec3(const Coord& ijk)
        : mVec{T(ijk[0]), T(ijk[1]), T(ijk[2])}
    {
    }
    __hostdev__ bool operator==(const Vec3& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2]; }
    __hostdev__ bool operator!=(const Vec3& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2]; }
    template<typename Vec3T>
    __hostdev__ Vec3& operator=(const Vec3T& rhs)
    {
        mVec[0] = rhs[0];
        mVec[1] = rhs[1];
        mVec[2] = rhs[2];
        return *this;
    }
    __hostdev__ const T& operator[](int i) const { return mVec[i]; }
    __hostdev__ T& operator[](int i) { return mVec[i]; }
    template<typename Vec3T>
    __hostdev__ T dot(const Vec3T& v) const { return mVec[0] * v[0] + mVec[1] * v[1] + mVec[2] * v[2]; }
    template<typename Vec3T>
    __hostdev__ Vec3 cross(const Vec3T& v) const
    {
        return Vec3(mVec[1] * v[2] - mVec[2] * v[1],
                    mVec[2] * v[0] - mVec[0] * v[2],
                    mVec[0] * v[1] - mVec[1] * v[0]);
    }
    __hostdev__ T lengthSqr() const
    {
        return mVec[0] * mVec[0] + mVec[1] * mVec[1] + mVec[2] * mVec[2]; // 5 flops
    }
    __hostdev__ T    length() const { return Sqrt(this->lengthSqr()); }
    __hostdev__ Vec3 operator-() const { return Vec3(-mVec[0], -mVec[1], -mVec[2]); }
    __hostdev__ Vec3 operator*(const Vec3& v) const { return Vec3(mVec[0] * v[0], mVec[1] * v[1], mVec[2] * v[2]); }
    __hostdev__ Vec3 operator/(const Vec3& v) const { return Vec3(mVec[0] / v[0], mVec[1] / v[1], mVec[2] / v[2]); }
    __hostdev__ Vec3 operator+(const Vec3& v) const { return Vec3(mVec[0] + v[0], mVec[1] + v[1], mVec[2] + v[2]); }
    __hostdev__ Vec3 operator-(const Vec3& v) const { return Vec3(mVec[0] - v[0], mVec[1] - v[1], mVec[2] - v[2]); }
    __hostdev__ Vec3 operator*(const T& s) const { return Vec3(s * mVec[0], s * mVec[1], s * mVec[2]); }
    __hostdev__ Vec3 operator/(const T& s) const { return (T(1) / s) * (*this); }
    __hostdev__ Vec3& operator+=(const Vec3& v)
    {
        mVec[0] += v[0];
        mVec[1] += v[1];
        mVec[2] += v[2];
        return *this;
    }
    __hostdev__ Vec3& operator-=(const Vec3& v)
    {
        mVec[0] -= v[0];
        mVec[1] -= v[1];
        mVec[2] -= v[2];
        return *this;
    }
    __hostdev__ Vec3& operator*=(const T& s)
    {
        mVec[0] *= s;
        mVec[1] *= s;
        mVec[2] *= s;
        return *this;
    }
    __hostdev__ Vec3& operator/=(const T& s) { return (*this) *= T(1) / s; }
    __hostdev__ Vec3& normalize() { return (*this) /= this->length(); }
    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Vec3& minComponent(const Vec3& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Vec3& maxComponent(const Vec3& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        return *this;
    }
    /// @brief Return the smallest vector component
    __hostdev__ ValueType min() const
    {
        return mVec[0] < mVec[1] ? (mVec[0] < mVec[2] ? mVec[0] : mVec[2]) : (mVec[1] < mVec[2] ? mVec[1] : mVec[2]);
    }
    /// @brief Return the largest vector component
    __hostdev__ ValueType max() const
    {
        return mVec[0] > mVec[1] ? (mVec[0] > mVec[2] ? mVec[0] : mVec[2]) : (mVec[1] > mVec[2] ? mVec[1] : mVec[2]);
    }
    __hostdev__ Coord floor() const { return Coord(Floor(mVec[0]), Floor(mVec[1]), Floor(mVec[2])); }
    __hostdev__ Coord ceil() const { return Coord(Ceil(mVec[0]), Ceil(mVec[1]), Ceil(mVec[2])); }
    __hostdev__ Coord round() const { return Coord(Floor(mVec[0] + 0.5), Floor(mVec[1] + 0.5), Floor(mVec[2] + 0.5)); }
}; // Vec3<T>

template<typename T1, typename T2>
__hostdev__ inline Vec3<T2> operator*(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar * vec[0], scalar * vec[1], scalar * vec[2]);
}
template<typename T1, typename T2>
__hostdev__ inline Vec3<T2> operator/(T1 scalar, const Vec3<T2>& vec)
{
    return Vec3<T2>(scalar / vec[0], scalar / vec[1], scalar / vec[2]);
}

using Vec3R = Vec3<double>;
using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;

/// @brief Return a single precision floating-point vector of this coordinate
__hostdev__ inline Vec3f Coord::asVec3s() const { return Vec3f(float(mVec[0]), float(mVec[1]), float(mVec[2])); }

/// @brief Return a double precision floating-point vector of this coordinate
__hostdev__ inline Vec3d Coord::asVec3d() const { return Vec3d(double(mVec[0]), double(mVec[1]), double(mVec[2])); }

// ----------------------------> Vec4 <--------------------------------------

/// @brief A simple vector class with three double components, similar to openvdb::math::Vec4
template<typename T>
class Vec4
{
    T mVec[4];

public:
    static const int SIZE = 4;
    using ValueType = T;
    Vec4() = default;
    __hostdev__ explicit Vec4(T x)
        : mVec{x, x, x, x}
    {
    }
    __hostdev__ Vec4(T x, T y, T z, T w)
        : mVec{x, y, z, w}
    {
    }
    template<typename T2>
    __hostdev__ explicit Vec4(const Vec4<T2>& v)
        : mVec{T(v[0]), T(v[1]), T(v[2]), T(v[3])}
    {
    }
    __hostdev__ bool operator==(const Vec4& rhs) const { return mVec[0] == rhs[0] && mVec[1] == rhs[1] && mVec[2] == rhs[2] && mVec[3] == rhs[3]; }
    __hostdev__ bool operator!=(const Vec4& rhs) const { return mVec[0] != rhs[0] || mVec[1] != rhs[1] || mVec[2] != rhs[2] || mVec[3] != rhs[3]; }
    template<typename Vec4T>
    __hostdev__ Vec4& operator=(const Vec4T& rhs)
    {
        mVec[0] = rhs[0];
        mVec[1] = rhs[1];
        mVec[2] = rhs[2];
        mVec[3] = rhs[3];
        return *this;
    }
    __hostdev__ const T& operator[](int i) const { return mVec[i]; }
    __hostdev__ T& operator[](int i) { return mVec[i]; }
    template<typename Vec4T>
    __hostdev__ T dot(const Vec4T& v) const { return mVec[0] * v[0] + mVec[1] * v[1] + mVec[2] * v[2] + mVec[3] * v[3]; }
    __hostdev__ T lengthSqr() const
    {
        return mVec[0] * mVec[0] + mVec[1] * mVec[1] + mVec[2] * mVec[2] + mVec[3] * mVec[3]; // 7 flops
    }
    __hostdev__ T    length() const { return Sqrt(this->lengthSqr()); }
    __hostdev__ Vec4 operator-() const { return Vec4(-mVec[0], -mVec[1], -mVec[2], -mVec[3]); }
    __hostdev__ Vec4 operator*(const Vec4& v) const { return Vec4(mVec[0] * v[0], mVec[1] * v[1], mVec[2] * v[2], mVec[3] * v[3]); }
    __hostdev__ Vec4 operator/(const Vec4& v) const { return Vec4(mVec[0] / v[0], mVec[1] / v[1], mVec[2] / v[2], mVec[3] / v[3]); }
    __hostdev__ Vec4 operator+(const Vec4& v) const { return Vec4(mVec[0] + v[0], mVec[1] + v[1], mVec[2] + v[2], mVec[3] + v[3]); }
    __hostdev__ Vec4 operator-(const Vec4& v) const { return Vec4(mVec[0] - v[0], mVec[1] - v[1], mVec[2] - v[2], mVec[3] - v[3]); }
    __hostdev__ Vec4 operator*(const T& s) const { return Vec4(s * mVec[0], s * mVec[1], s * mVec[2], s * mVec[3]); }
    __hostdev__ Vec4 operator/(const T& s) const { return (T(1) / s) * (*this); }
    __hostdev__ Vec4& operator+=(const Vec4& v)
    {
        mVec[0] += v[0];
        mVec[1] += v[1];
        mVec[2] += v[2];
        mVec[3] += v[3];
        return *this;
    }
    __hostdev__ Vec4& operator-=(const Vec4& v)
    {
        mVec[0] -= v[0];
        mVec[1] -= v[1];
        mVec[2] -= v[2];
        mVec[3] -= v[3];
        return *this;
    }
    __hostdev__ Vec4& operator*=(const T& s)
    {
        mVec[0] *= s;
        mVec[1] *= s;
        mVec[2] *= s;
        mVec[3] *= s;
        return *this;
    }
    __hostdev__ Vec4& operator/=(const T& s) { return (*this) *= T(1) / s; }
    __hostdev__ Vec4& normalize() { return (*this) /= this->length(); }
    /// @brief Perform a component-wise minimum with the other Coord.
    __hostdev__ Vec4& minComponent(const Vec4& other)
    {
        if (other[0] < mVec[0])
            mVec[0] = other[0];
        if (other[1] < mVec[1])
            mVec[1] = other[1];
        if (other[2] < mVec[2])
            mVec[2] = other[2];
        if (other[3] < mVec[3])
            mVec[3] = other[3];
        return *this;
    }

    /// @brief Perform a component-wise maximum with the other Coord.
    __hostdev__ Vec4& maxComponent(const Vec4& other)
    {
        if (other[0] > mVec[0])
            mVec[0] = other[0];
        if (other[1] > mVec[1])
            mVec[1] = other[1];
        if (other[2] > mVec[2])
            mVec[2] = other[2];
        if (other[3] > mVec[3])
            mVec[3] = other[3];
        return *this;
    }
}; // Vec4<T>

template<typename T1, typename T2>
__hostdev__ inline Vec4<T2> operator*(T1 scalar, const Vec4<T2>& vec)
{
    return Vec4<T2>(scalar * vec[0], scalar * vec[1], scalar * vec[2], scalar * vec[3]);
}
template<typename T1, typename T2>
__hostdev__ inline Vec4<T2> operator/(T1 scalar, const Vec3<T2>& vec)
{
    return Vec4<T2>(scalar / vec[0], scalar / vec[1], scalar / vec[2], scalar / vec[3]);
}

using Vec4R = Vec4<double>;
using Vec4d = Vec4<double>;
using Vec4f = Vec4<float>;
using Vec4i = Vec4<int>;

// ----------------------------> TensorTraits <--------------------------------------

template<typename T, int Rank = (is_specialization<T, Vec3>::value ||
                                 is_specialization<T, Vec4>::value ||
                                 is_same<T, Rgba8>::value) ? 1 : 0>
struct TensorTraits;

template<typename T>
struct TensorTraits<T, 0>
{
    static const int  Rank = 0; // i.e. scalar
    static const bool IsScalar = true;
    static const bool IsVector = false;
    static const int  Size = 1;
    using ElementType = T;
    static T scalar(const T& s) { return s; }
};

template<typename T>
struct TensorTraits<T, 1>
{
    static const int  Rank = 1; // i.e. vector
    static const bool IsScalar = false;
    static const bool IsVector = true;
    static const int  Size = T::SIZE;
    using ElementType = typename T::ValueType;
    static ElementType scalar(const T& v) { return v.length(); }
};

// ----------------------------> FloatTraits <--------------------------------------

template<typename T, int = sizeof(typename TensorTraits<T>::ElementType)>
struct FloatTraits
{
    using FloatType = float;
};

template<typename T>
struct FloatTraits<T, 8>
{
    using FloatType = double;
};

template<>
struct FloatTraits<bool, 1>
{
    using FloatType = bool;
};

template<>
struct FloatTraits<ValueMask, 1>
{
    using FloatType = bool;
};

// ----------------------------> mapping ValueType -> GridType <--------------------------------------

/// @brief Maps from a templated value type to a GridType enum
template<typename BuildT>
__hostdev__ inline GridType mapToGridType()
{
    if (is_same<BuildT, float>::value) { // resolved at compile-time
        return GridType::Float;
    } else if (is_same<BuildT, double>::value) {
        return GridType::Double;
    } else if (is_same<BuildT, int16_t>::value) {
        return GridType::Int16;
    } else if (is_same<BuildT, int32_t>::value) {
        return GridType::Int32;
    } else if (is_same<BuildT, int64_t>::value) {
        return GridType::Int64;
    } else if (is_same<BuildT, Vec3f>::value) {
        return GridType::Vec3f;
    } else if (is_same<BuildT, Vec3d>::value) {
        return GridType::Vec3d;
    } else if (is_same<BuildT, uint32_t>::value) {
        return GridType::UInt32;
    } else if (is_same<BuildT, ValueMask>::value) {
        return GridType::Mask;
    } else if (is_same<BuildT, bool>::value) {
        return GridType::Boolean;
    } else if (is_same<BuildT, Rgba8>::value) {
        return GridType::RGBA8;
    } else if (is_same<BuildT, Fp4>::value) {
        return GridType::Fp4;
    } else if (is_same<BuildT, Fp8>::value) {
        return GridType::Fp8;
    } else if (is_same<BuildT, Fp16>::value) {
        return GridType::Fp16;
    } else if (is_same<BuildT, FpN>::value) {
        return GridType::FpN;
    } else if (is_same<BuildT, Vec4f>::value) {
        return GridType::Vec4f;
    } else if (is_same<BuildT, Vec4d>::value) {
        return GridType::Vec4d;
    }
    return GridType::Unknown;
}

// ----------------------------> matMult <--------------------------------------

template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[1], xyz[2] * mat[2])),
                 fmaf(xyz[0], mat[3], fmaf(xyz[1], mat[4], xyz[2] * mat[5])),
                 fmaf(xyz[0], mat[6], fmaf(xyz[1], mat[7], xyz[2] * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], static_cast<double>(xyz[2]) * mat[2])),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[5])),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[1], fmaf(xyz[2], mat[2], vec[0]))),
                 fmaf(xyz[0], mat[3], fmaf(xyz[1], mat[4], fmaf(xyz[2], mat[5], vec[1]))),
                 fmaf(xyz[0], mat[6], fmaf(xyz[1], mat[7], fmaf(xyz[2], mat[8], vec[2])))); // 9 fmaf = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMult(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[1], fma(static_cast<double>(xyz[2]), mat[2], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[3], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[5], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[6], fma(static_cast<double>(xyz[1]), mat[7], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}

// matMultT: Multiply with the transpose:

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const float* mat, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[3], xyz[2] * mat[6])),
                 fmaf(xyz[0], mat[1], fmaf(xyz[1], mat[4], xyz[2] * mat[7])),
                 fmaf(xyz[0], mat[2], fmaf(xyz[1], mat[5], xyz[2] * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const double* mat, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], static_cast<double>(xyz[2]) * mat[6])),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], static_cast<double>(xyz[2]) * mat[7])),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], static_cast<double>(xyz[2]) * mat[8]))); // 6 fmaf + 3 mult = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const float* mat, const float* vec, const Vec3T& xyz)
{
    return Vec3T(fmaf(xyz[0], mat[0], fmaf(xyz[1], mat[3], fmaf(xyz[2], mat[6], vec[0]))),
                 fmaf(xyz[0], mat[1], fmaf(xyz[1], mat[4], fmaf(xyz[2], mat[7], vec[1]))),
                 fmaf(xyz[0], mat[2], fmaf(xyz[1], mat[5], fmaf(xyz[2], mat[8], vec[2])))); // 9 fmaf = 9 flops
}

template<typename Vec3T>
__hostdev__ inline Vec3T matMultT(const double* mat, const double* vec, const Vec3T& xyz)
{
    return Vec3T(fma(static_cast<double>(xyz[0]), mat[0], fma(static_cast<double>(xyz[1]), mat[3], fma(static_cast<double>(xyz[2]), mat[6], vec[0]))),
                 fma(static_cast<double>(xyz[0]), mat[1], fma(static_cast<double>(xyz[1]), mat[4], fma(static_cast<double>(xyz[2]), mat[7], vec[1]))),
                 fma(static_cast<double>(xyz[0]), mat[2], fma(static_cast<double>(xyz[1]), mat[5], fma(static_cast<double>(xyz[2]), mat[8], vec[2])))); // 9 fma = 9 flops
}

// ----------------------------> BBox <-------------------------------------

// Base-class for static polymorphism (cannot be constructed directly)
template<typename Vec3T>
struct BaseBBox
{
    Vec3T             mCoord[2];
    __hostdev__ bool  operator==(const BaseBBox& rhs) const { return mCoord[0] == rhs.mCoord[0] && mCoord[1] == rhs.mCoord[1]; };
    __hostdev__ bool  operator!=(const BaseBBox& rhs) const { return mCoord[0] != rhs.mCoord[0] || mCoord[1] != rhs.mCoord[1]; };
    __hostdev__ const Vec3T& operator[](int i) const { return mCoord[i]; }
    __hostdev__ Vec3T& operator[](int i) { return mCoord[i]; }
    __hostdev__ Vec3T& min() { return mCoord[0]; }
    __hostdev__ Vec3T& max() { return mCoord[1]; }
    __hostdev__ const Vec3T& min() const { return mCoord[0]; }
    __hostdev__ const Vec3T& max() const { return mCoord[1]; }
    __hostdev__ Coord&       translate(const Vec3T& xyz)
    {
        mCoord[0] += xyz;
        mCoord[1] += xyz;
        return *this;
    }
    // @brief Expand this bounding box to enclose point (i, j, k).
    __hostdev__ BaseBBox& expand(const Vec3T& xyz)
    {
        mCoord[0].minComponent(xyz);
        mCoord[1].maxComponent(xyz);
        return *this;
    }
    //__hostdev__ BaseBBox expandBy(typename Vec3T::ValueType padding) const
    //{
    //    return BaseBBox(mCoord[0].offsetBy(-padding),mCoord[1].offsetBy(padding));
    //}
    __hostdev__ bool isInside(const Vec3T& xyz)
    {
        if (xyz[0] < mCoord[0][0] || xyz[1] < mCoord[0][1] || xyz[2] < mCoord[0][2])
            return false;
        if (xyz[0] > mCoord[1][0] || xyz[1] > mCoord[1][1] || xyz[2] > mCoord[1][2])
            return false;
        return true;
    }

protected:
    __hostdev__ BaseBBox() {}
    __hostdev__ BaseBBox(const Vec3T& min, const Vec3T& max)
        : mCoord{min, max}
    {
    }
}; // BaseBBox

template<typename Vec3T, bool = is_floating_point<typename Vec3T::ValueType>::value>
struct BBox;

/// @brief Partial template specialization for floating point coordinate types.
///
/// @note Min is inclusive and max is exclusive. If min = max the dimension of
///       the bounding box is zero and therefore it is also empty.
template<typename Vec3T>
struct BBox<Vec3T, true> : public BaseBBox<Vec3T>
{
    using Vec3Type = Vec3T;
    using ValueType = typename Vec3T::ValueType;
    static_assert(is_floating_point<ValueType>::value, "Expected a floating point coordinate type");
    using BaseT = BaseBBox<Vec3T>;
    using BaseT::mCoord;
    __hostdev__ BBox()
        : BaseT(Vec3T( Maximum<typename Vec3T::ValueType>::value()),
                Vec3T(-Maximum<typename Vec3T::ValueType>::value()))
    {
    }
    __hostdev__ BBox(const Vec3T& min, const Vec3T& max)
        : BaseT(min, max)
    {
    }
    __hostdev__ BBox(const Coord& min, const Coord& max)
        : BaseT(Vec3T(ValueType(min[0]), ValueType(min[1]), ValueType(min[2])),
                Vec3T(ValueType(max[0] + 1), ValueType(max[1] + 1), ValueType(max[2] + 1)))
    {
    }
    __hostdev__ BBox(const BaseBBox<Coord>& bbox) : BBox(bbox[0], bbox[1]) {}
    __hostdev__ bool  empty() const { return mCoord[0][0] >= mCoord[1][0] ||
                                             mCoord[0][1] >= mCoord[1][1] ||
                                             mCoord[0][2] >= mCoord[1][2]; }
    __hostdev__ Vec3T dim() const { return this->empty() ? Vec3T(0) : this->max() - this->min(); }
    __hostdev__ bool  isInside(const Vec3T& p) const
    {
        return p[0] > mCoord[0][0] && p[1] > mCoord[0][1] && p[2] > mCoord[0][2] &&
               p[0] < mCoord[1][0] && p[1] < mCoord[1][1] && p[2] < mCoord[1][2];
    }
};// BBox<Vec3T, true>

/// @brief Partial template specialization for integer coordinate types
///
/// @note Both min and max are INCLUDED in the bbox so dim = max - min + 1. So,
///       if min = max the bounding box contains exactly one point and dim = 1!
template<typename CoordT>
struct BBox<CoordT, false> : public BaseBBox<CoordT>
{
    static_assert(is_same<int, typename CoordT::ValueType>::value, "Expected \"int\" coordinate type");
    using BaseT = BaseBBox<CoordT>;
    using BaseT::mCoord;
    /// @brief Iterator over the domain covered by a BBox
    /// @details z is the fastest-moving coordinate.
    class Iterator
    {
        const BBox& mBBox;
        CoordT      mPos;
    public:
        __hostdev__ Iterator(const BBox& b)
            : mBBox(b)
            , mPos(b.min())
        {
        }
        __hostdev__ Iterator& operator++()
        {
            if (mPos[2] < mBBox[1][2]) {// this is the most common case
                ++mPos[2];
            } else if (mPos[1] < mBBox[1][1]) {
                mPos[2] = mBBox[0][2];
                ++mPos[1];
            } else if (mPos[0] <= mBBox[1][0]) {
                mPos[2] = mBBox[0][2];
                mPos[1] = mBBox[0][1];
                ++mPos[0];
            }
            return *this;
        }
        __hostdev__ Iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        /// @brief Return @c true if the iterator still points to a valid coordinate.
        __hostdev__       operator bool() const { return mPos[0] <= mBBox[1][0]; }
        __hostdev__ const CoordT& operator*() const { return mPos; }
    }; // Iterator
    __hostdev__ Iterator begin() const { return Iterator{*this}; }
    __hostdev__          BBox()
        : BaseT(CoordT::max(), CoordT::min())
    {
    }
    __hostdev__ BBox(const CoordT& min, const CoordT& max)
        : BaseT(min, max)
    {
    }
    template<typename SplitT>
    __hostdev__ BBox(BBox& other, const SplitT&)
        : BaseT(other.mCoord[0], other.mCoord[1])
    {
        NANOVDB_ASSERT(this->is_divisible());
        const int n = MaxIndex(this->dim());
        mCoord[1][n] = (mCoord[0][n] + mCoord[1][n]) >> 1;
        other.mCoord[0][n] = mCoord[1][n] + 1;
    }
    __hostdev__ bool is_divisible() const { return mCoord[0][0] < mCoord[1][0] &&
                                                   mCoord[0][1] < mCoord[1][1] &&
                                                   mCoord[0][2] < mCoord[1][2]; }
    /// @brief Return true if this bounding box is empty, i.e. uninitialized
    __hostdev__ bool   empty() const { return mCoord[0][0] > mCoord[1][0] ||
                                              mCoord[0][1] > mCoord[1][1] ||
                                              mCoord[0][2] > mCoord[1][2]; }
    __hostdev__ CoordT dim() const { return this->empty() ? Coord(0) : this->max() - this->min() + Coord(1); }
    __hostdev__ uint64_t volume() const { auto d = this->dim(); return uint64_t(d[0])*uint64_t(d[1])*uint64_t(d[2]); }
    __hostdev__ bool   isInside(const CoordT& p) const { return !(CoordT::lessThan(p, this->min()) || CoordT::lessThan(this->max(), p)); }
    __hostdev__ bool   isInside(const BBox& b) const
    {
        return !(CoordT::lessThan(b.min(), this->min()) || CoordT::lessThan(this->max(), b.max()));
    }

    /// @warning This converts a CoordBBox into a floating-point bounding box which implies that max += 1 !
    template<typename RealT>
    __hostdev__ BBox<Vec3<RealT>> asReal() const
    {
        static_assert(is_floating_point<RealT>::value, "CoordBBox::asReal: Expected a floating point coordinate");
        return BBox<Vec3<RealT>>(Vec3<RealT>(RealT(mCoord[0][0]), RealT(mCoord[0][1]), RealT(mCoord[0][2])),
                                 Vec3<RealT>(RealT(mCoord[1][0] + 1), RealT(mCoord[1][1] + 1), RealT(mCoord[1][2] + 1)));
    }
    /// @brief Return a new instance that is expanded by the specified padding.
    __hostdev__ BBox expandBy(typename CoordT::ValueType padding) const
    {
        return BBox(mCoord[0].offsetBy(-padding), mCoord[1].offsetBy(padding));
    }
};// BBox<CoordT, false>

using CoordBBox = BBox<Coord>;
using BBoxR = BBox<Vec3R>;

// -------------------> Find lowest and highest bit in a word <----------------------------

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 32 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ static inline uint32_t FindLowestOn(uint32_t v)
{
    NANOVDB_ASSERT(v);
#if defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanForward(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzl(v));
#else
    static const unsigned char DeBruijn[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint32_t((v & -v) * 0x077CB531U) >> 27];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 32 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ static inline uint32_t FindHighestOn(uint32_t v)
{
    NANOVDB_ASSERT(v);
#if defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanReverse(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __builtin_clzl(v);

#else
    static const unsigned char DeBruijn[32] = {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31};
    v |= v >> 1; // first round down to one less than a power of 2
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return DeBruijn[uint32_t(v * 0x07C4ACDDU) >> 27];
#endif
}

/// @brief Returns the index of the lowest, i.e. least significant, on bit in the specified 64 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ static inline uint32_t FindLowestOn(uint64_t v)
{
    NANOVDB_ASSERT(v);
#if defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanForward64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return static_cast<uint32_t>(__builtin_ctzll(v));
#else
    static const unsigned char DeBruijn[64] = {
        0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
        62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
        63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
        51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
    };
// disable unary minus on unsigned warning
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(push)
#pragma warning(disable : 4146)
#endif
    return DeBruijn[uint64_t((v & -v) * UINT64_C(0x022FDD63CC95386D)) >> 58];
#if defined(_MSC_VER) && !defined(__NVCC__)
#pragma warning(pop)
#endif

#endif
}

/// @brief Returns the index of the highest, i.e. most significant, on bit in the specified 64 bit word
///
/// @warning Assumes that at least one bit is set in the word, i.e. @a v != uint32_t(0)!
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ static inline uint32_t FindHighestOn(uint64_t v)
{
    NANOVDB_ASSERT(v);
#if defined(_MSC_VER) && defined(NANOVDB_USE_INTRINSICS)
    unsigned long index;
    _BitScanReverse64(&index, v);
    return static_cast<uint32_t>(index);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(NANOVDB_USE_INTRINSICS)
    return sizeof(unsigned long) * 8 - 1 - __builtin_clzll(v);
#else
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&v);
    return p[1] ? 32u + FindHighestOn(p[1]) : FindHighestOn(p[0]);
#endif
}

// ----------------------------> CountOn <--------------------------------------

/// @return Number of bits that are on in the specified 64-bit word
NANOVDB_HOSTDEV_DISABLE_WARNING
__hostdev__ inline uint32_t CountOn(uint64_t v)
{
#if defined(_MSC_VER) && defined(_M_X64)
    v = __popcnt64(v);
#elif (defined(__GNUC__) || defined(__clang__))
    v = __builtin_popcountll(v);
#else
    // Software Implementation
    v = v - ((v >> 1) & uint64_t(0x5555555555555555));
    v = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    v = (((v + (v >> 4)) & uint64_t(0xF0F0F0F0F0F0F0F)) * uint64_t(0x101010101010101)) >> 56;
#endif
    return static_cast<uint32_t>(v);
}

// ----------------------------> Mask <--------------------------------------

/// @brief Bit-mask to encode active states and facilitate sequential iterators
/// and a fast codec for I/O compression.
template<uint32_t LOG2DIM>
class Mask
{
    static constexpr uint32_t SIZE = 1U << (3 * LOG2DIM); // Number of bits in mask
    static constexpr uint32_t WORD_COUNT = SIZE >> 6; // Number of 64 bit words
    uint64_t                  mWords[WORD_COUNT];

public:
    /// @brief Return the memory footprint in bytes of this Mask
    __hostdev__ static size_t memUsage() { return sizeof(Mask); }

    /// @brief Return the number of bits available in this Mask
    __hostdev__ static uint32_t bitCount() { return SIZE; }

    /// @brief Return the number of machine words used by this Mask
    __hostdev__ static uint32_t wordCount() { return WORD_COUNT; }

    __hostdev__ uint32_t countOn() const
    {
        uint32_t sum = 0, n = WORD_COUNT;
        for (const uint64_t* w = mWords; n--; ++w)
            sum += CountOn(*w);
        return sum;
    }

    class Iterator
    {
    public:
        __hostdev__ Iterator()
            : mPos(Mask::SIZE)
            , mParent(nullptr)
        {
        }
        __hostdev__ Iterator(uint32_t pos, const Mask* parent)
            : mPos(pos)
            , mParent(parent)
        {
        }
        Iterator&            operator=(const Iterator&) = default;
        __hostdev__ uint32_t operator*() const { return mPos; }
        __hostdev__          operator bool() const { return mPos != Mask::SIZE; }
        __hostdev__ Iterator& operator++()
        {
            mPos = mParent->findNextOn(mPos + 1);
            return *this;
        }

    private:
        uint32_t    mPos;
        const Mask* mParent;
    }; // Member class MaskIterator

    /// @brief Initialize all bits to zero.
    __hostdev__ Mask()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = 0;
    }
    __hostdev__ Mask(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = v;
    }

    /// @brief Copy constructor
    __hostdev__ Mask(const Mask& other)
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = other.mWords[i];
    }

    /// @brief Return the <i>n</i>th word of the bit mask, for a word of arbitrary size.
    template<typename WordT>
    __hostdev__ WordT getWord(int n) const
    {
        NANOVDB_ASSERT(n * 8 * sizeof(WordT) < SIZE);
        return reinterpret_cast<const WordT*>(mWords)[n];
    }

    /// @brief Assignment operator that works with openvdb::util::NodeMask
    template<typename MaskT>
    __hostdev__ Mask& operator=(const MaskT& other)
    {
        static_assert(sizeof(Mask) == sizeof(MaskT), "Mismatching sizeof");
        static_assert(WORD_COUNT == MaskT::WORD_COUNT, "Mismatching word count");
        static_assert(LOG2DIM == MaskT::LOG2DIM, "Mismatching LOG2DIM");
        auto     *src = reinterpret_cast<const uint64_t*>(&other);
        uint64_t *dst = mWords;
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            *dst++ = *src++;
        }
        return *this;
    }

    __hostdev__ bool operator==(const Mask& other) const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i) {
            if (mWords[i] != other.mWords[i]) return false;
        }
        return true;
    }

    __hostdev__ bool operator!=(const Mask& other) const { return !((*this) == other); }

    __hostdev__ Iterator beginOn() const { return Iterator(this->findFirstOn(), this); }

    /// @brief Return true if the given bit is set.
    __hostdev__ bool isOn(uint32_t n) const { return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); }

    __hostdev__ bool isOn() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            if (mWords[i] != ~uint64_t(0))
                return false;
        return true;
    }

    __hostdev__ bool isOff() const
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            if (mWords[i] != uint64_t(0))
                return false;
        return true;
    }

    /// @brief Set the given bit on.
    __hostdev__ void setOn(uint32_t n) { mWords[n >> 6] |= uint64_t(1) << (n & 63); }
    __hostdev__ void setOff(uint32_t n) { mWords[n >> 6] &= ~(uint64_t(1) << (n & 63)); }

    __hostdev__ void set(uint32_t n, bool On)
    {
#if 1   // switch between branchless
        auto &word = mWords[n >> 6];
        n &= 63;
        word &= ~(uint64_t(1) << n);
        word |=   uint64_t(On) << n;
#else
        On ? this->setOn(n) : this->setOff(n);
#endif
    }

    /// @brief Set all bits on
    __hostdev__ void setOn()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = ~uint64_t(0);
    }

    /// @brief Set all bits off
    __hostdev__ void setOff()
    {
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = uint64_t(0);
    }

    /// @brief Set all bits off
    __hostdev__ void set(bool on)
    {
        const uint64_t v = on ? ~uint64_t(0) : uint64_t(0);
        for (uint32_t i = 0; i < WORD_COUNT; ++i)
            mWords[i] = v;
    }
    /// brief Toggle the state of all bits in the mask
    __hostdev__ void toggle()
    {
        uint32_t n = WORD_COUNT;
        for (auto* w = mWords; n--; ++w)
            *w = ~*w;
    }
    __hostdev__ void toggle(uint32_t n) { mWords[n >> 6] ^= uint64_t(1) << (n & 63); }

private:

    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint32_t findFirstOn() const
    {
        uint32_t        n = 0;
        const uint64_t* w = mWords;
        for (; n < WORD_COUNT && !*w; ++w, ++n)
            ;
        return n == WORD_COUNT ? SIZE : (n << 6) + FindLowestOn(*w);
    }

    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint32_t findNextOn(uint32_t start) const
    {
        uint32_t n = start >> 6; // initiate
        if (n >= WORD_COUNT)
            return SIZE; // check for out of bounds
        uint32_t m = start & 63;
        uint64_t b = mWords[n];
        if (b & (uint64_t(1) << m))
            return start; // simple case: start is on
        b &= ~uint64_t(0) << m; // mask out lower bits
        while (!b && ++n < WORD_COUNT)
            b = mWords[n]; // find next non-zero word
        return (!b ? SIZE : (n << 6) + FindLowestOn(b)); // catch last word=0
    }
}; // Mask class

// ----------------------------> Map <--------------------------------------

/// @brief Defines an affine transform and its inverse represented as a 3x3 matrix and a vec3 translation
struct Map
{
    float  mMatF[9]; // 9*4B <- 3x3 matrix
    float  mInvMatF[9]; // 9*4B <- 3x3 matrix
    float  mVecF[3]; // 3*4B <- translation
    float  mTaperF; // 4B, placeholder for taper value
    double mMatD[9]; // 9*8B <- 3x3 matrix
    double mInvMatD[9]; // 9*8B <- 3x3 matrix
    double mVecD[3]; // 3*8B <- translation
    double mTaperD; // 8B, placeholder for taper value

    // This method can only be called on the host to initialize the member data
    template<typename Mat4T>
    __hostdev__ void set(const Mat4T& mat, const Mat4T& invMat, double taper);

    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return matMult(mMatD, mVecD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return matMult(mMatF, mVecF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return matMult(mMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return matMult(mMatF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const
    {
        return matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const
    {
        return matMult(mInvMatF, Vec3T(xyz[0] - mVecF[0], xyz[1] - mVecF[1], xyz[2] - mVecF[2]));
    }

    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return matMult(mInvMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return matMult(mInvMatF, xyz); }

    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return matMultT(mInvMatD, xyz); }
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return matMultT(mInvMatF, xyz); }
}; // Map

template<typename Mat4T>
__hostdev__ void Map::set(const Mat4T& mat, const Mat4T& invMat, double taper)
{
    float * mf = mMatF, *vf = mVecF;
    float*  mif = mInvMatF;
    double *md = mMatD, *vd = mVecD;
    double* mid = mInvMatD;
    mTaperF = static_cast<float>(taper);
    mTaperD = taper;
    for (int i = 0; i < 3; ++i) {
        *vd++ = mat[3][i]; //translation
        *vf++ = static_cast<float>(mat[3][i]);
        for (int j = 0; j < 3; ++j) {
            *md++ = mat[j][i]; //transposed
            *mid++ = invMat[j][i];
            *mf++ = static_cast<float>(mat[j][i]);
            *mif++ = static_cast<float>(invMat[j][i]);
        }
    }
}

// ----------------------------> GridBlindMetaData <--------------------------------------

struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridBlindMetaData
{
    static const int      MaxNameSize = 256;// due to NULL termination the maximum length is one less!
    int64_t               mByteOffset; // byte offset to the blind data, relative to the GridData.
    uint64_t              mElementCount; // number of elements, e.g. point count
    uint32_t              mFlags; // flags
    GridBlindDataSemantic mSemantic; // semantic meaning of the data.
    GridBlindDataClass    mDataClass; // 4 bytes
    GridType              mDataType; // 4 bytes
    char                  mName[MaxNameSize];// note this include the NULL termination

    /// @brief return memory usage in bytes for the class (note this computes for all blindMetaData structures.)
    __hostdev__ static uint64_t memUsage(uint64_t blindDataCount = 0)
    {
        return blindDataCount * sizeof(GridBlindMetaData);
    }

    __hostdev__  void setBlindData(void *ptr) { mByteOffset = PtrDiff(ptr, this); }

    template <typename T>
    __hostdev__ const T* getBlindData() const { return PtrAdd<T>(this, mByteOffset); }

}; // GridBlindMetaData

// ----------------------------> NodeTrait <--------------------------------------

/// @brief Struct to derive node type from its level in a given
///        grid, tree or root while perserving constness
template<typename GridOrTreeOrRootT, int LEVEL>
struct NodeTrait;

// Partial template specialization of above Node struct
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 0>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::LeafNodeType;
    using type = typename GridOrTreeOrRootT::LeafNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 0>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::LeafNodeType;
    using type = const typename GridOrTreeOrRootT::LeafNodeType;
};

template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 1>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootType::ChildNodeType::ChildNodeType;
    using type = typename GridOrTreeOrRootT::RootType::ChildNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 1>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootType::ChildNodeType::ChildNodeType;
    using type = const typename GridOrTreeOrRootT::RootType::ChildNodeType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 2>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootType::ChildNodeType;
    using type = typename GridOrTreeOrRootT::RootType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 2>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootType::ChildNodeType;
    using type = const typename GridOrTreeOrRootT::RootType::ChildNodeType;
};
template<typename GridOrTreeOrRootT>
struct NodeTrait<GridOrTreeOrRootT, 3>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = typename GridOrTreeOrRootT::RootType;
    using type = typename GridOrTreeOrRootT::RootType;
};

template<typename GridOrTreeOrRootT>
struct NodeTrait<const GridOrTreeOrRootT, 3>
{
    static_assert(GridOrTreeOrRootT::RootType::LEVEL == 3, "Tree depth is not supported");
    using Type = const typename GridOrTreeOrRootT::RootType;
    using type = const typename GridOrTreeOrRootT::RootType;
};

// ----------------------------> Grid <--------------------------------------

/*
    The following class and comment is for internal use only

    Memory layout:

    Grid ->       39 x double                          (world bbox and affine transformation)
    Tree -> Root  3 x ValueType + int32_t + N x Tiles  (background,min,max,tileCount + tileCount x Tiles)

    N2 upper InternalNodes each with 2 bit masks, N2 tiles, and min/max values

    N1 lower InternalNodes each with 2 bit masks, N1 tiles, and min/max values

    N0 LeafNodes each with a bit mask, N0 ValueTypes and min/max

    Example layout: ("---" implies it has a custom offset, "..." implies zero or more)
    [GridData][TreeData]---[RootData][ROOT TILES...]---[NodeData<5>]---[ModeData<4>]---[LeafData<3>]---[BLINDMETA...]---[BLIND0]---[BLIND1]---etc.
*/

/// @brief Struct with all the member data of the Grid (useful during serialization of an openvdb grid)
///
/// @note The transform is assumed to be affine (so linear) and have uniform scale! So frustum transforms
///       and non-uniform scaling are not supported (primarily because they complicate ray-tracing in index space)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) GridData
{// sizeof(GridData) = 672B
    static const int MaxNameSize = 256;// due to NULL termination the maximum length is one less
    uint64_t         mMagic; // 8B magic to validate it is valid grid data.
    uint64_t         mChecksum; // 8B. Checksum of grid buffer.
    Version          mVersion;// 4B major, minor, and patch version numbers
    uint32_t         mFlags; // 4B. flags for grid.
    uint32_t         mGridIndex;// 4B. Index of this grid in the buffer
    uint32_t         mGridCount; // 4B. Total number of grids in the buffer
    uint64_t         mGridSize; // 8B. byte count of this entire grid occupied in the buffer.
    char             mGridName[MaxNameSize]; // 256B
    Map              mMap; // 264B. affine transformation between index and world space in both single and double precision
    BBox<Vec3R>      mWorldBBox; // 48B. floating-point AABB of active values in WORLD SPACE (2 x 3 doubles)
    Vec3R            mVoxelSize; // 24B. size of a voxel in world units
    GridClass        mGridClass; // 4B.
    GridType         mGridType; //  4B.
    int64_t          mBlindMetadataOffset; // 8B. offset of GridBlindMetaData structures that follow this grid.
    uint32_t         mBlindMetadataCount; // 4B. count of GridBlindMetaData structures that follow this grid.


    // Set and unset various bit flags
    __hostdev__ void setFlagsOff() { mFlags = uint32_t(0); }
    __hostdev__ void setMinMaxOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::HasMinMax);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::HasMinMax);
        }
    }
    __hostdev__ void setBBoxOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::HasBBox);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::HasBBox);
        }
    }
    __hostdev__ void setLongGridNameOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::HasLongGridName);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::HasLongGridName);
        }
    }
    __hostdev__ void setAverageOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::HasAverage);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::HasAverage);
        }
    }
    __hostdev__ void setStdDeviationOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::HasStdDeviation);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::HasStdDeviation);
        }
    }
    __hostdev__ void setBreadthFirstOn(bool on = true)
    {
        if (on) {
            mFlags |= static_cast<uint32_t>(GridFlags::IsBreadthFirst);
        } else {
            mFlags &= ~static_cast<uint32_t>(GridFlags::IsBreadthFirst);
        }
    }

    // Affine transformations based on double precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMap(const Vec3T& xyz) const { return mMap.applyMap(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMap(const Vec3T& xyz) const { return mMap.applyInverseMap(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobian(const Vec3T& xyz) const { return mMap.applyJacobian(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobian(const Vec3T& xyz) const { return mMap.applyInverseJacobian(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJT(const Vec3T& xyz) const { return mMap.applyIJT(xyz); }
    // Affine transformations based on single precision
    template<typename Vec3T>
    __hostdev__ Vec3T applyMapF(const Vec3T& xyz) const { return mMap.applyMapF(xyz); } // Pos: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseMapF(const Vec3T& xyz) const { return mMap.applyInverseMapF(xyz); } // Pos: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyJacobianF(const Vec3T& xyz) const { return mMap.applyJacobianF(xyz); } // Dir: index -> world
    template<typename Vec3T>
    __hostdev__ Vec3T applyInverseJacobianF(const Vec3T& xyz) const { return mMap.applyInverseJacobianF(xyz); } // Dir: world -> index
    template<typename Vec3T>
    __hostdev__ Vec3T applyIJTF(const Vec3T& xyz) const { return mMap.applyIJTF(xyz); }

    // @brief Return a non-const void pointer to the tree
    __hostdev__ void* treePtr() { return this + 1; }

    // @brief Return a const void pointer to the tree
    __hostdev__ const void* treePtr() const { return this + 1; }

    /// @brief Returns a const reference to the blindMetaData at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ const GridBlindMetaData* blindMetaData(uint32_t n) const
    {
        NANOVDB_ASSERT(n < mBlindMetadataCount);
        return PtrAdd<GridBlindMetaData>(this, mBlindMetadataOffset) + n;
    }

}; // GridData

// Forward declaration of accelerated random access class
template <typename BuildT, int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1>
class ReadAccessor;

template <typename BuildT>
using DefaultReadAccessor = ReadAccessor<BuildT, 0, 1, 2>;

/// @brief Highest level of the data structure. Contains a tree and a world->index
///        transform (that currently only supports uniform scaling and translation).
///
/// @note This the API of this class to interface with client code
template<typename TreeT>
class Grid : private GridData
{
public:
    using TreeType  = TreeT;
    using RootType  = typename TreeT::RootType;
    using DataType  = GridData;
    using ValueType = typename TreeT::ValueType;
    using BuildType = typename TreeT::BuildType;// in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordType = typename TreeT::CoordType;
    using AccessorType = DefaultReadAccessor<BuildType>;

    /// @brief Disallow constructions, copy and assignment
    ///
    /// @note Only a Serializer, defined elsewhere, can instantiate this class
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    ~Grid() = delete;

    __hostdev__ Version version() const { return DataType::mVersion; }

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return memory usage in bytes for this class only.
    __hostdev__ static uint64_t memUsage() { return sizeof(GridData); }

    /// @brief Return the memory footprint of the entire grid, i.e. including all nodes and blind data
    __hostdev__ uint64_t gridSize() const { return DataType::mGridSize; }

    /// @brief Return index of this grid in the buffer
    __hostdev__ uint32_t gridIndex() const { return DataType::mGridIndex; }

    /// @brief Return total number of grids in the buffer
    __hostdev__ uint32_t gridCount() const { return DataType::mGridCount; }

    /// @brief Return a const reference to the tree
    __hostdev__ const TreeT& tree() const { return *reinterpret_cast<const TreeT*>(this->treePtr()); }

    /// @brief Return a non-const reference to the tree
    __hostdev__ TreeT& tree() { return *reinterpret_cast<TreeT*>(this->treePtr()); }

    /// @brief Return a new instance of a ReadAccessor used to access values in this grid
    __hostdev__ AccessorType getAccessor() const { return AccessorType(this->tree().root()); }

    /// @brief Return a const reference to the size of a voxel in world units
    __hostdev__ const Vec3R& voxelSize() const { return DataType::mVoxelSize; }

    /// @brief Return a const reference to the Map for this grid
    __hostdev__ const Map& map() const { return DataType::mMap; }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndex(const Vec3T& xyz) const { return this->applyInverseMap(xyz); }

    /// @brief index to world space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorld(const Vec3T& xyz) const { return this->applyMap(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDir(const Vec3T& dir) const { return this->applyJacobian(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDir(const Vec3T& dir) const { return this->applyInverseJacobian(dir); }

    /// @brief transform the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGrad(const Vec3T& grad) const { return this->applyIJT(grad); }

    /// @brief world to index space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexF(const Vec3T& xyz) const { return this->applyInverseMapF(xyz); }

    /// @brief index to world space transformation
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldF(const Vec3T& xyz) const { return this->applyMapF(xyz); }

    /// @brief transformation from index space direction to world space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldDirF(const Vec3T& dir) const { return this->applyJacobianF(dir); }

    /// @brief transformation from world space direction to index space direction
    /// @warning assumes dir to be normalized
    template<typename Vec3T>
    __hostdev__ Vec3T worldToIndexDirF(const Vec3T& dir) const { return this->applyInverseJacobianF(dir); }

    /// @brief Transforms the gradient from index space to world space.
    /// @details Applies the inverse jacobian transform map.
    template<typename Vec3T>
    __hostdev__ Vec3T indexToWorldGradF(const Vec3T& grad) const { return DataType::applyIJTF(grad); }

    /// @brief Computes a AABB of active values in world space
    __hostdev__ const BBox<Vec3R>& worldBBox() const { return DataType::mWorldBBox; }

    /// @brief Computes a AABB of active values in index space
    ///
    /// @note This method is returning a floating point bounding box and not a CoordBBox. This makes
    ///       it more useful for clipping rays.
    __hostdev__ const BBox<CoordType>& indexBBox() const { return this->tree().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ uint64_t activeVoxelCount() const { return this->tree().activeVoxelCount(); }

    /// @brief Methods related to the classification of this grid
    __hostdev__ bool  isValid() const { return DataType::mMagic == NANOVDB_MAGIC_NUMBER; }
    __hostdev__ const GridType&  gridType() const { return DataType::mGridType; }
    __hostdev__ const GridClass& gridClass() const { return DataType::mGridClass; }
    __hostdev__ bool             isLevelSet() const { return DataType::mGridClass == GridClass::LevelSet; }
    __hostdev__ bool             isFogVolume() const { return DataType::mGridClass == GridClass::FogVolume; }
    __hostdev__ bool             isStaggered() const { return DataType::mGridClass == GridClass::Staggered; }
    __hostdev__ bool             isPointIndex() const { return DataType::mGridClass == GridClass::PointIndex; }
    __hostdev__ bool             isPointData() const { return DataType::mGridClass == GridClass::PointData; }
    __hostdev__ bool             isMask() const { return DataType::mGridClass == GridClass::Topology; }
    __hostdev__ bool             isUnknown() const { return DataType::mGridClass == GridClass::Unknown; }
    __hostdev__ bool             hasMinMax() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::HasMinMax); }
    __hostdev__ bool             hasBBox() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::HasBBox); }
    __hostdev__ bool             hasLongGridName() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::HasLongGridName); }
    __hostdev__ bool             hasAverage() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::HasAverage); }
    __hostdev__ bool             hasStdDeviation() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::HasStdDeviation); }
    __hostdev__ bool             isBreadthFirst() const { return DataType::mFlags & static_cast<uint32_t>(GridFlags::IsBreadthFirst); }

    /// @brief return true if the specified node type is layed out breadth-first in memory and has a fixed size.
    ///        This allows for sequential access to the nodes.
    template <typename NodeT>
    __hostdev__ bool isSequential() const { return NodeT::FIXED_SIZE && this->isBreadthFirst(); }

    /// @brief return true if the specified node level is layed out breadth-first in memory and has a fixed size.
    ///        This allows for sequential access to the nodes.
    template <int LEVEL>
    __hostdev__ bool isSequential() const { return NodeTrait<TreeT,LEVEL>::type::FIXED_SIZE && this->isBreadthFirst(); }

    /// @brief Return a c-string with the name of this grid
    __hostdev__ const char* gridName() const
    {
        if (this->hasLongGridName()) {
            const auto &metaData = this->blindMetaData(DataType::mBlindMetadataCount-1);// always the last
            NANOVDB_ASSERT(metaData.mDataClass == GridBlindDataClass::GridName);
            return metaData.template getBlindData<const char>();
        }
        return DataType::mGridName;
    }

    /// @brief Return a c-string with the name of this grid, truncated to 255 characters
    __hostdev__ const char* shortGridName() const { return DataType::mGridName; }

    /// @brief Return checksum of the grid buffer.
    __hostdev__ uint64_t checksum() const { return DataType::mChecksum; }

    /// @brief Return true if this grid is empty, i.e. contains no values or nodes.
    __hostdev__ bool isEmpty() const { return this->tree().isEmpty(); }

    /// @brief Return the count of blind-data encoded in this grid
    __hostdev__ int blindDataCount() const { return DataType::mBlindMetadataCount; }

    /// @brief Return the index of the blind data with specified semantic if found, otherwise -1.
    __hostdev__ int findBlindDataForSemantic(GridBlindDataSemantic semantic) const;

    /// @brief Returns a const pointer to the blindData at the specified linear offset.
    ///
    /// @warning Point might be NULL and the linear offset is assumed to be in the valid range
    __hostdev__ const void* blindData(uint32_t n) const
    {
        if (DataType::mBlindMetadataCount == 0) {
            return nullptr;
        }
        NANOVDB_ASSERT(n < DataType::mBlindMetadataCount);
        return this->blindMetaData(n).template getBlindData<void>();
    }

    __hostdev__ const GridBlindMetaData& blindMetaData(int n) const { return *DataType::blindMetaData(n); }

private:
    static_assert(sizeof(GridData) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(GridData) is misaligned");
}; // Class Grid

template<typename TreeT>
__hostdev__ int Grid<TreeT>::findBlindDataForSemantic(GridBlindDataSemantic semantic) const
{
    for (uint32_t i = 0, n = this->blindDataCount(); i < n; ++i)
        if (this->blindMetaData(i).mSemantic == semantic)
            return int(i);
    return -1;
}

// ----------------------------> Tree <--------------------------------------

template<int ROOT_LEVEL = 3>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) TreeData
{// sizeof(TreeData<3>) == 64B
    static_assert(ROOT_LEVEL == 3, "Root level is assumed to be three");
    uint64_t mNodeOffset[4];//32B, byte offset from this tree to first leaf, lower, upper and root node
    uint32_t mNodeCount[3];// 12B, total number of nodes of type: leaf, lower internal, upper internal
    uint32_t mTileCount[3];// 12B, total number of tiles of type: leaf, lower internal, upper internal (node, only active tiles!)
    uint64_t mVoxelCount;//    8B, total number of active voxels in the root and all its child nodes.

    template <typename RootT>
    __hostdev__ void setRoot(const RootT* root) { mNodeOffset[3] = PtrDiff(root, this); }
    template <typename RootT>
    __hostdev__ RootT* getRoot() { return PtrAdd<RootT>(this, mNodeOffset[3]); }
    template <typename RootT>
    __hostdev__ const RootT* getRoot() const { return PtrAdd<RootT>(this, mNodeOffset[3]); }

    template <typename NodeT>
    __hostdev__ void setFirstNode(const NodeT* node)
    {
        mNodeOffset[NodeT::LEVEL] = node ? PtrDiff(node, this) : 0;
    }
};

// ----------------------------> GridTree <--------------------------------------

/// @brief defines a tree type from a grid type while perserving constness
template<typename GridT>
struct GridTree
{
    using Type = typename GridT::TreeType;
    using type = typename GridT::TreeType;
};
template<typename GridT>
struct GridTree<const GridT>
{
    using Type = const typename GridT::TreeType;
    using type = const typename GridT::TreeType;
};

// ----------------------------> Tree <--------------------------------------

/// @brief VDB Tree, which is a thin wrapper around a RootNode.
template<typename RootT>
class Tree : private TreeData<RootT::LEVEL>
{
    static_assert(RootT::LEVEL == 3, "Tree depth is not supported");
    static_assert(RootT::ChildNodeType::LOG2DIM == 5, "Tree configuration is not supported");
    static_assert(RootT::ChildNodeType::ChildNodeType::LOG2DIM == 4, "Tree configuration is not supported");
    static_assert(RootT::LeafNodeType::LOG2DIM == 3, "Tree configuration is not supported");

public:
    using DataType = TreeData<RootT::LEVEL>;
    using RootType = RootT;
    using LeafNodeType = typename RootT::LeafNodeType;
    using ValueType = typename RootT::ValueType;
    using BuildType = typename RootT::BuildType;// in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordType = typename RootT::CoordType;
    using AccessorType = DefaultReadAccessor<BuildType>;

    using Node3 = RootT;
    using Node2 = typename RootT::ChildNodeType;
    using Node1 = typename Node2::ChildNodeType;
    using Node0 = LeafNodeType;

    /// @brief This class cannot be constructed or deleted
    Tree() = delete;
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    ~Tree() = delete;

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static uint64_t memUsage() { return sizeof(DataType); }

    __hostdev__ RootT& root() { return *DataType::template getRoot<RootT>(); }

    __hostdev__ const RootT& root() const { return *DataType::template getRoot<RootT>(); }

    __hostdev__ AccessorType getAccessor() const { return AccessorType(this->root()); }

    /// @brief Return the value of the given voxel (regardless of state or location in the tree.)
    __hostdev__ ValueType getValue(const CoordType& ijk) const { return this->root().getValue(ijk); }

    /// @brief Return the active state of the given voxel (regardless of state or location in the tree.)
    __hostdev__ bool isActive(const CoordType& ijk) const { return this->root().isActive(ijk); }

    /// @brief Return true if this tree is empty, i.e. contains no values or nodes
    __hostdev__ bool isEmpty() const { return this->root().isEmpty(); }

    /// @brief Combines the previous two methods in a single call
    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const { return this->root().probeValue(ijk, v); }

    /// @brief Return a const reference to the background value.
    __hostdev__ const ValueType& background() const { return this->root().background(); }

    /// @brief Sets the extrema values of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ void extrema(ValueType& min, ValueType& max) const;

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ const BBox<CoordType>& bbox() const { return this->root().bbox(); }

    /// @brief Return the total number of active voxels in this tree.
    __hostdev__ uint64_t activeVoxelCount() const { return DataType::mVoxelCount; }

    /// @brief   Return the total number of active tiles at the specified level of the tree.
    ///
    /// @details n = 0 corresponds to leaf level tiles.
    __hostdev__ const uint32_t& activeTileCount(uint32_t n) const
    {
        NANOVDB_ASSERT(n < 3);
        return DataType::mTileCount[n];
    }

    template<typename NodeT>
    __hostdev__ uint32_t nodeCount() const
    {
        static_assert(NodeT::LEVEL < 3, "Invalid NodeT");
        return DataType::mNodeCount[NodeT::LEVEL];
    }

    __hostdev__ uint32_t nodeCount(int level) const
    {
        NANOVDB_ASSERT(level < 3);
        return DataType::mNodeCount[level];
    }

    /// @brief return a pointer to the first node of the specified type
    ///
    /// @warning Note it may return NULL if no nodes exist
     template <typename NodeT>
    __hostdev__ NodeT* getFirstNode()
    {
        const uint64_t offset = DataType::mNodeOffset[NodeT::LEVEL];
        return offset>0 ? PtrAdd<NodeT>(this, offset) : nullptr;
    }

    /// @brief return a const pointer to the first node of the specified type
    ///
    /// @warning Note it may return NULL if no nodes exist
    template <typename NodeT>
    __hostdev__ const NodeT* getFirstNode() const
    {
        const uint64_t offset = DataType::mNodeOffset[NodeT::LEVEL];
        return offset>0 ? PtrAdd<NodeT>(this, offset) : nullptr;
    }

    /// @brief return a pointer to the first node at the specified level
    ///
    /// @warning Note it may return NULL if no nodes exist
    template <int LEVEL>
    __hostdev__ typename NodeTrait<RootT, LEVEL>::type*
    getFirstNode()
    {
        return this->template getFirstNode<typename NodeTrait<RootT,LEVEL>::type>();
    }

    /// @brief return a const pointer to the first node of the specified level
    ///
    /// @warning Note it may return NULL if no nodes exist
    template <int LEVEL>
    __hostdev__ const typename NodeTrait<RootT, LEVEL>::type*
    getFirstNode() const
    {
        return this->template getFirstNode<typename NodeTrait<RootT,LEVEL>::type>();
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(TreeData) is misaligned");

}; // Tree class

template<typename RootT>
__hostdev__ void Tree<RootT>::extrema(ValueType& min, ValueType& max) const
{
    min = this->root().minimum();
    max = this->root().maximum();
}

// --------------------------> RootNode <------------------------------------

/// @brief Struct with all the member data of the RootNode (useful during serialization of an openvdb RootNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) RootData
{
    using ValueT = typename ChildT::ValueType;
    using BuildT = typename ChildT::BuildType;// in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using CoordT = typename ChildT::CoordType;
    using StatsT = typename ChildT::FloatType;
    static constexpr bool FIXED_SIZE = false;

    /// @brief Return a key based on the coordinates of a voxel
#ifdef USE_SINGLE_ROOT_KEY
    using KeyT = uint64_t;
    template <typename CoordType>
    __hostdev__ static KeyT CoordToKey(const CoordType& ijk)
    {
        static_assert(sizeof(CoordT) == sizeof(CoordType), "Mismatching sizeof");
        static_assert(32 - ChildT::TOTAL <= 21, "Cannot use 64 bit root keys");
        return (KeyT(uint32_t(ijk[2]) >> ChildT::TOTAL)) | //       z is the lower 21 bits
               (KeyT(uint32_t(ijk[1]) >> ChildT::TOTAL) << 21) | // y is the middle 21 bits
               (KeyT(uint32_t(ijk[0]) >> ChildT::TOTAL) << 42); //  x is the upper 21 bits
    }
    __hostdev__ static CoordT KeyToCoord(const KeyT& key)
    {
        static constexpr uint64_t MASK = (1u << 21) - 1;
        return CoordT(((key >> 42) & MASK) << ChildT::TOTAL,
                      ((key >> 21) & MASK) << ChildT::TOTAL,
                       (key & MASK) << ChildT::TOTAL);
    }
#else
    using KeyT = CoordT;
    __hostdev__ static KeyT   CoordToKey(const CoordT& ijk) { return ijk & ~ChildT::MASK; }
    __hostdev__ static CoordT KeyToCoord(const KeyT& key) { return key; }
#endif
    BBox<CoordT> mBBox; // 24B. AABB if active values in index space.
    uint32_t     mTableSize; // 4B. number of tiles and child pointers in the root node

    ValueT mBackground; // background value, i.e. value of any unset voxel
    ValueT mMinimum; // typically 4B, minmum of all the active values
    ValueT mMaximum; // typically 4B, maximum of all the active values
    StatsT mAverage; // typically 4B, average of all the active values in this node and its child nodes
    StatsT mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes

    struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) Tile
    {
        template <typename CoordType>
        __hostdev__ void setChild(const CoordType& k, const ChildT *ptr, const RootData *data)
        {
            key = CoordToKey(k);
            child = PtrDiff(ptr, data);
        }
        template <typename CoordType, typename ValueType>
        __hostdev__ void setValue(const CoordType& k, bool s, const ValueType &v)
        {
            key = CoordToKey(k);
            state = s;
            value = v;
            child = 0;
        }
        __hostdev__ bool  isChild() const { return child; }
        __hostdev__ CoordT origin() const { return KeyToCoord(key); }
        KeyT               key; // USE_SINGLE_ROOT_KEY ? 8B : 12B
        int64_t            child; // 8B. signed byte offset from this node to the child node.  0 means it is a constant tile, so use value.
        uint32_t           state; // 4B. state of tile value
        ValueT             value; // value of tile (i.e. no child node)
    }; // Tile

    /// @brief Returns a non-const reference to the tile at the specified linear offset.
    ///
    /// @warning The linear offset is assumed to be in the valid range
    __hostdev__ const Tile* tile(uint32_t n) const
    {
        NANOVDB_ASSERT(n < mTableSize);
        return reinterpret_cast<const Tile*>(this + 1) + n;
    }
    __hostdev__ Tile* tile(uint32_t n)
    {
        NANOVDB_ASSERT(n < mTableSize);
        return reinterpret_cast<Tile*>(this + 1) + n;
    }

    /// @brief Returns a const reference to the child node in the specified tile.
    ///
    /// @warning A child node is assumed to exist in the specified tile
    __hostdev__ ChildT* getChild(const Tile* tile)
    {
        NANOVDB_ASSERT(tile->child);
        return PtrAdd<ChildT>(this, tile->child);
    }
    __hostdev__ const ChildT* getChild(const Tile* tile) const
    {
        NANOVDB_ASSERT(tile->child);
        return PtrAdd<ChildT>(this, tile->child);
    }

    __hostdev__ const ValueT& getMin()     const { return mMinimum; }
    __hostdev__ const ValueT& getMax()     const { return mMaximum; }
    __hostdev__ const StatsT& average()      const { return mAverage; }
    __hostdev__ const StatsT& stdDeviation() const { return mStdDevi; }

    __hostdev__ void setMin(const ValueT& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueT& v) { mMaximum = v; }
    __hostdev__ void setAvg(const StatsT& v) { mAverage = v; }
    __hostdev__ void setDev(const StatsT& v) { mStdDevi = v; }

    /// @brief This class cannot be constructed or deleted
    RootData() = delete;
    RootData(const RootData&) = delete;
    RootData& operator=(const RootData&) = delete;
    ~RootData() = delete;
}; // RootData

/// @brief Top-most node of the VDB tree structure.
template<typename ChildT>
class RootNode : private RootData<ChildT>
{
public:
    using DataType = RootData<ChildT>;
    using LeafNodeType = typename ChildT::LeafNodeType;
    using ChildNodeType = ChildT;
    using RootType = RootNode<ChildT>;// this allows RootNode to behave like a Tree

    using ValueType = typename DataType::ValueT;
    using FloatType = typename DataType::StatsT;
    using BuildType = typename DataType::BuildT;// in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool

    using CoordType = typename ChildT::CoordType;
    using AccessorType = DefaultReadAccessor<BuildType>;
    using Tile = typename DataType::Tile;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;

    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf

    /// @brief This class cannot be constructed or deleted
    RootNode() = delete;
    RootNode(const RootNode&) = delete;
    RootNode& operator=(const RootNode&) = delete;
    ~RootNode() = delete;

    __hostdev__ AccessorType getAccessor() const { return AccessorType(*this); }

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the index bounding box of all the active values in this tree, i.e. in all nodes of the tree
    __hostdev__ const BBox<CoordType>& bbox() const { return DataType::mBBox; }

    /// @brief Return the total number of active voxels in the root and all its child nodes.

    /// @brief Return a const reference to the background value, i.e. the value associated with
    ///        any coordinate location that has not been set explicitly.
    __hostdev__ const ValueType& background() const { return DataType::mBackground; }

    /// @brief Return the number of tiles encoded in this root node
    __hostdev__ const uint32_t& tileCount() const { return DataType::mTableSize; }

    /// @brief Return a const reference to the minimum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& minimum() const { return this->getMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this root node and any of its child nodes
    __hostdev__ const ValueType& maximum() const { return this->getMax(); }

    /// @brief Return a const reference to the average of all the active values encoded in this root node and any of its child nodes
    __hostdev__ const FloatType& average() const { return DataType::mAverage; }

    /// @brief Return the variance of all the active values encoded in this root node and any of its child nodes
    __hostdev__ FloatType variance() const { return DataType::mStdDevi * DataType::mStdDevi; }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this root node and any of its child nodes
    __hostdev__ const FloatType& stdDeviation() const { return DataType::mStdDevi; }

    /// @brief Return the expected memory footprint in bytes with the specified number of tiles
    __hostdev__ static uint64_t memUsage(uint32_t tableSize) { return sizeof(RootNode) + tableSize * sizeof(Tile); }

    /// @brief Return the actual memory footprint of this root node
    __hostdev__ uint64_t memUsage() const { return sizeof(RootNode) + DataType::mTableSize * sizeof(Tile); }

    /// @brief Return the value of the given voxel
    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            return tile->isChild() ? this->getChild(tile)->getValue(ijk) : tile->value;
        }
        return DataType::mBackground;
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            return tile->isChild() ? this->getChild(tile)->isActive(ijk) : tile->state;
        }
        return false;
    }

    /// @brief Return true if this RootNode is empty, i.e. contains no values or nodes
    __hostdev__ bool isEmpty() const { return DataType::mTableSize == uint32_t(0); }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->isChild()) {
                const auto *child = this->getChild(tile);
                return child->probeValue(ijk, v);
            }
            v = tile->value;
            return tile->state;
        }
        v = DataType::mBackground;
        return false;
    }

    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const
    {
        const Tile* tile = this->findTile(ijk);
        if (tile && tile->isChild()) {
            const auto *child = this->getChild(tile);
            return child->probeLeaf(ijk);
        }
        return nullptr;
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData) is misaligned");
    static_assert(sizeof(typename DataType::Tile) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(RootData::Tile) is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class Tree;

    /// @brief Private method to find a Tile of this root node by means of binary-search. This is obviously
    ///        much slower then direct lookup into a linear array (as in the other nodes) which is exactly
    ///        why it is important to use the ReadAccessor which amortizes this overhead by node caching and
    ///        inverse tree traversal!
    __hostdev__ const Tile* findTile(const CoordType& ijk) const
    {
        const Tile* tiles = reinterpret_cast<const Tile*>(this + 1);
        const auto  key = DataType::CoordToKey(ijk);
#if 1   // switch between linear and binary seach
        for (uint32_t i = 0; i < DataType::mTableSize; ++i) {
            if (tiles[i].key == key) return &tiles[i];
        }
#else// do not enable binary search if tiles are not guaranteed to be sorted!!!!!!
        // binary-search of pre-sorted elements
        int32_t low = 0, high = DataType::mTableSize; // low is inclusive and high is exclusive
        while (low != high) {
            int         mid = low + ((high - low) >> 1);
            const Tile* tile = &tiles[mid];
            if (tile->key == key) {
                return tile;
            } else if (tile->key < key) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
#endif
        return nullptr;
    }

    /// @brief Private method to return node information and update a ReadAccessor
    template<typename AccT>
    __hostdev__ typename AccT::NodeInfo getNodeInfoAndCache(const CoordType& ijk, const AccT& acc) const
    {
        using NodeInfoT = typename AccT::NodeInfo;
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->isChild()) {
                const auto *child = this->getChild(tile);
                acc.insert(ijk, child);
                return child->getNodeInfoAndCache(ijk, acc);
            }
            return NodeInfoT{LEVEL, ChildT::dim(), tile->value, tile->value, tile->value,
                                 0, tile->origin(), tile->origin() + CoordType(ChildT::DIM)};
        }
        return NodeInfoT{LEVEL, ChildT::dim(), this->minimum(), this->maximum(),
                         this->average(), this->stdDeviation(), this->bbox()[0], this->bbox()[1]};
    }

    /// @brief Private method to return a voxel value and update a ReadAccessor
    template<typename AccT>
    __hostdev__ ValueType getValueAndCache(const CoordType& ijk, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->isChild()) {
                const auto *child = this->getChild(tile);
                acc.insert(ijk, child);
                return child->getValueAndCache(ijk, acc);
            }
            return tile->value;
        }
        return DataType::mBackground;
    }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const Tile* tile = this->findTile(ijk);
        if (tile && tile->isChild()) {
            const auto *child = this->getChild(tile);
            acc.insert(ijk, child);
            return child->isActiveAndCache(ijk, acc);
        }
        return false;
    }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordType& ijk, ValueType& v, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->isChild()) {
                const auto *child = this->getChild(tile);
                acc.insert(ijk, child);
                return child->probeValueAndCache(ijk, v, acc);
            }
            v = tile->value;
            return tile->state;
        }
        v = DataType::mBackground;
        return false;
    }

    template<typename AccT>
    __hostdev__ const LeafNodeType* probeLeafAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const Tile* tile = this->findTile(ijk);
        if (tile && tile->isChild()) {
            const auto *child = this->getChild(tile);
            acc.insert(ijk, child);
            return child->probeLeafAndCache(ijk, acc);
        }
        return nullptr;
    }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (const Tile* tile = this->findTile(ijk)) {
            if (tile->isChild()) {
                const auto *child = this->getChild(tile);
                acc.insert(ijk, child);
                return child->getDimAndCache(ijk, ray, acc);
            }
            return 1 << ChildT::TOTAL; //tile value
        }
        return ChildNodeType::dim(); // background
    }
}; // RootNode class

// After the RootNode the memory layout is assumed to be the sorted Tiles

// --------------------------> InternalNode <------------------------------------

/// @brief Struct with all the member data of the InternalNode (useful during serialization of an openvdb InternalNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ChildT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) InternalData
{
    using ValueT = typename ChildT::ValueType;
    using BuildT = typename ChildT::BuildType;// in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using StatsT = typename ChildT::FloatType;
    using CoordT = typename ChildT::CoordType;
    using MaskT  = typename ChildT::template MaskType<LOG2DIM>;
    static constexpr bool FIXED_SIZE = true;

    union Tile
    {
        ValueT  value;
        int64_t child;//signed 64 bit byte offset relative to the InternalData!!
        /// @brief This class cannot be constructed or deleted
        Tile() = delete;
        Tile(const Tile&) = delete;
        Tile& operator=(const Tile&) = delete;
        ~Tile() = delete;
    };

    BBox<CoordT> mBBox; // 24B. node bounding box.                   |
    uint64_t     mFlags; // 8B. node flags.                          | 32B aligned
    MaskT        mValueMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B  | 32B aligned
    MaskT        mChildMask; // LOG2DIM(5): 4096B, LOG2DIM(4): 512B  | 32B aligned

    ValueT mMinimum; // typically 4B
    ValueT mMaximum; // typically 4B
    StatsT mAverage; // typically 4B, average of all the active values in this node and its child nodes
    StatsT mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes
    alignas(32) Tile mTable[1u << (3 * LOG2DIM)]; // sizeof(ValueT) x (16*16*16 or 32*32*32)

    __hostdev__ void setChild(uint32_t n, const void *ptr)
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        mTable[n].child = PtrDiff(ptr, this);
    }

    template <typename ValueT>
    __hostdev__ void setValue(uint32_t n, const ValueT &v)
    {
        NANOVDB_ASSERT(!mChildMask.isOn(n));
        mTable[n].value = v;
    }

    /// @brief Returns a pointer to the child node at the specifed linear offset.
    __hostdev__ ChildT* getChild(uint32_t n)
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        return PtrAdd<ChildT>(this, mTable[n].child);
    }
    __hostdev__ const ChildT* getChild(uint32_t n) const
    {
        NANOVDB_ASSERT(mChildMask.isOn(n));
        return PtrAdd<ChildT>(this, mTable[n].child);
    }

    template <typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBox[0] = ijk; }

    __hostdev__ const ValueT& getMin()       const { return mMinimum; }
    __hostdev__ const ValueT& getMax()       const { return mMaximum; }
    __hostdev__ const StatsT& average()      const { return mAverage; }
    __hostdev__ const StatsT& stdDeviation() const { return mStdDevi; }

    __hostdev__ void setMin(const ValueT& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueT& v) { mMaximum = v; }
    __hostdev__ void setAvg(const StatsT& v) { mAverage = v; }
    __hostdev__ void setDev(const StatsT& v) { mStdDevi = v; }

    /// @brief This class cannot be constructed or deleted
    InternalData() = delete;
    InternalData(const InternalData&) = delete;
    InternalData& operator=(const InternalData&) = delete;
    ~InternalData() = delete;
}; // InternalData

/// @brief Internal nodes of a VDB treedim(),
template<typename ChildT, uint32_t Log2Dim = ChildT::LOG2DIM + 1>
class InternalNode : private InternalData<ChildT, Log2Dim>
{
public:
    using DataType = InternalData<ChildT, Log2Dim>;
    using ValueType = typename DataType::ValueT;
    using FloatType = typename DataType::StatsT;
    using BuildType = typename DataType::BuildT; // in rare cases BuildType != ValueType, e.g. then BuildType = ValueMask and ValueType = bool
    using LeafNodeType = typename ChildT::LeafNodeType;
    using ChildNodeType = ChildT;
    using CoordType = typename ChildT::CoordType;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;
    template<uint32_t LOG2>
    using MaskType = typename ChildT::template MaskType<LOG2>;

    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; // dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL; // number of voxels along each axis of this node
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); // number of tile values (or child pointers)
    static constexpr uint32_t MASK = (1u << TOTAL) - 1u;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node

    /// @brief This class cannot be constructed or deleted
    InternalNode() = delete;
    InternalNode(const InternalNode&) = delete;
    InternalNode& operator=(const InternalNode&) = delete;
    ~InternalNode() = delete;

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return the dimension, in voxel units, of this internal node (typically 8*16 or 8*16*32)
    __hostdev__ static uint32_t dim() { return 1u << TOTAL; }

    /// @brief Return memory usage in bytes for the class
    __hostdev__ static size_t memUsage() { return sizeof(DataType); }

    /// @brief Return a const reference to the bit mask of active voxels in this internal node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }

    /// @brief Return a const reference to the bit mask of child nodes in this internal node
    __hostdev__ const MaskType<LOG2DIM>& childMask() const { return DataType::mChildMask; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordType origin() const { return DataType::mBBox.min() & ~MASK; }

    /// @brief Return a const reference to the minimum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& minimum() const { return this->getMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this internal node and any of its child nodes
    __hostdev__ const ValueType& maximum() const { return this->getMax(); }

    /// @brief Return a const reference to the average of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ const FloatType& average() const { return DataType::mAverage; }

    /// @brief Return the variance of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ FloatType variance() const { return DataType::mStdDevi*DataType::mStdDevi; }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this internal node and any of its child nodes
    __hostdev__ const FloatType& stdDeviation() const { return DataType::mStdDevi; }

    /// @brief Return a const reference to the bounding box in index space of active values in this internal node and any of its child nodes
    __hostdev__ const BBox<CoordType>& bbox() const { return DataType::mBBox; }

    /// @brief Return the value of the given voxel
    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->getChild(n)->getValue(ijk) : DataType::mTable[n].value;
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        return DataType::mChildMask.isOn(n) ? this->getChild(n)->isActive(ijk) : DataType::mValueMask.isOn(n);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n))
            return this->getChild(n)->probeValue(ijk, v);
        v = DataType::mTable[n].value;
        return DataType::mValueMask.isOn(n);
    }

    __hostdev__ const LeafNodeType* probeLeaf(const CoordType& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n))
            return this->getChild(n)->probeLeaf(ijk);
        return nullptr;
    }

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordType& ijk)
    {
#if 0
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
               ((ijk[2] & MASK) >> ChildT::TOTAL);
#else
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) |
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) |
               ((ijk[2] & MASK) >> ChildT::TOTAL);
#endif
    }

    /// @return the local coordinate of the n'th tile or child node
    __hostdev__ static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & ((1 << LOG2DIM) - 1));
    }

    /// @brief modifies local coordinates to global coordinates of a tile or child node
    __hostdev__ void localToGlobalCoord(Coord& ijk) const
    {
        ijk <<= ChildT::TOTAL;
        ijk += this->origin();
    }

    __hostdev__ Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = InternalNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    /// @brief Retrun true if this node or any of its child nodes contain active values
    __hostdev__ bool isActive() const
    {
        return DataType::mFlags & uint32_t(2);
    }

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(InternalData) is misaligned");
    //static_assert(offsetof(DataType, mTable) % 32 == 0, "InternalData::mTable is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    /// @brief Private read access method used by the ReadAccessor
    template<typename AccT>
    __hostdev__ ValueType getValueAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return DataType::mTable[n].value;
        const ChildT* child = this->getChild(n);
        acc.insert(ijk, child);
        return child->getValueAndCache(ijk, acc);
    }

    template<typename AccT>
    __hostdev__ typename AccT::NodeInfo getNodeInfoAndCache(const CoordType& ijk, const AccT& acc) const
    {
        using NodeInfoT = typename AccT::NodeInfo;
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n)) {
            return NodeInfoT{LEVEL, this->dim(), this->minimum(), this->maximum(), this->average(),
                             this->stdDeviation(), this->bbox()[0], this->bbox()[1]};
        }
        const ChildT* child = this->getChild(n);
        acc.insert(ijk, child);
        return child->getNodeInfoAndCache(ijk, acc);
    }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return DataType::mValueMask.isOn(n);
        const ChildT* child = this->getChild(n);
        acc.insert(ijk, child);
        return child->isActiveAndCache(ijk, acc);
    }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordType& ijk, ValueType& v, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n)) {
            v = DataType::mTable[n].value;
            return DataType::mValueMask.isOn(n);
        }
        const ChildT* child = this->getChild(n);
        acc.insert(ijk, child);
        return child->probeValueAndCache(ijk, v, acc);
    }

    template<typename AccT>
    __hostdev__ const LeafNodeType* probeLeafAndCache(const CoordType& ijk, const AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (!DataType::mChildMask.isOn(n))
            return nullptr;
        const ChildT* child = this->getChild(n);
        acc.insert(ijk, child);
        return child->probeLeafAndCache(ijk, acc);
    }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordType& ijk, const RayT& ray, const AccT& acc) const
    {
        if (DataType::mFlags & uint32_t(1))
            this->dim(); //ship this node if first bit is set
        //if (!ray.intersects( this->bbox() )) return 1<<TOTAL;

        const uint32_t n = CoordToOffset(ijk);
        if (DataType::mChildMask.isOn(n)) {
            const ChildT* child = this->getChild(n);
            acc.insert(ijk, child);
            return child->getDimAndCache(ijk, ray, acc);
        }
        return ChildNodeType::dim(); // tile value
    }

}; // InternalNode class

// --------------------------> LeafNode <------------------------------------

/// @brief Stuct with all the member data of the LeafNode (useful during serialization of an openvdb LeafNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename ValueT, typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = ValueT;
    using BuildType = ValueT;
    using FloatType = typename FloatTraits<ValueT>::FloatType;
    using ArrayType = ValueT;// type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B.
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    ValueType mMinimum; // typically 4B
    ValueType mMaximum; // typically 4B
    FloatType mAverage; // typically 4B, average of all the active values in this node and its child nodes
    FloatType mStdDevi; // typically 4B, standard deviation of all the active values in this node and its child nodes
    alignas(32) ValueType mValues[1u << 3 * LOG2DIM];

    //__hostdev__ const ValueType* values() const { return mValues; }
    __hostdev__ ValueType getValue(uint32_t i) const { return mValues[i]; }
    __hostdev__ void setValueOnly(uint32_t offset, const ValueType& value) { mValues[offset] = value; }
    __hostdev__ void setValue(uint32_t offset, const ValueType& value)
    {
        mValueMask.setOn(offset);
        mValues[offset] = value;
    }

    __hostdev__ ValueType getMin() const { return mMinimum; }
    __hostdev__ ValueType getMax() const { return mMaximum; }
    __hostdev__ FloatType getAvg() const { return mAverage; }
    __hostdev__ FloatType getDev() const { return mStdDevi; }

    __hostdev__ void setMin(const ValueType& v) { mMinimum = v; }
    __hostdev__ void setMax(const ValueType& v) { mMaximum = v; }
    __hostdev__ void setAvg(const FloatType& v) { mAverage = v; }
    __hostdev__ void setDev(const FloatType& v) { mStdDevi = v; }

    template <typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<ValueT>

/// @brief Base-class for quantized float leaf nodes
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafFnBase
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = float;
    using FloatType = float;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B.
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    float mMinimum; //  4B - minimum of ALL values in this node
    float mQuantum; //  = (max - min)/15 4B
    uint16_t mMin, mMax, mAvg, mDev;// quantized representations of statistics of active values

    void init(float min, float max, uint8_t bitWidth)
    {
        mMinimum = min;
        mQuantum = (max - min)/float((1 << bitWidth)-1);
    }

    /// @brief return the quantized minimum of the active values in this node
    __hostdev__ float getMin()    const { return mMin*mQuantum + mMinimum; }

    /// @brief return the quantized maximum of the active values in this node
    __hostdev__ float getMax()    const { return mMax*mQuantum + mMinimum; }

    /// @brief return the quantized average of the active values in this node
    __hostdev__ float getAvg()    const { return mAvg*mQuantum + mMinimum; }
    /// @brief return the quantized standard deviation of the active values in this node

    /// @note 0 <= StdDev <= max-min or 0 <= StdDev/(max-min) <= 1
    __hostdev__ float getDev() const { return mDev*mQuantum; }

    /// @note min <= X <= max or 0 <= (X-min)/(min-max) <= 1
    __hostdev__ void setMin(float min) { mMin = uint16_t((min - mMinimum)/mQuantum + 0.5f); }

    /// @note min <= X <= max or 0 <= (X-min)/(min-max) <= 1
    __hostdev__ void setMax(float max) { mMax = uint16_t((max - mMinimum)/mQuantum + 0.5f); }

    /// @note min <= avg <= max or 0 <= (avg-min)/(min-max) <= 1
    __hostdev__ void setAvg(float avg) { mAvg = uint16_t((avg - mMinimum)/mQuantum + 0.5f); }

    /// @note 0 <= StdDev <= max-min or 0 <= StdDev/(max-min) <= 1
    __hostdev__ void setDev(float dev) { mDev = uint16_t(dev/mQuantum + 0.5f); }

    template <typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }
};// LeafFnBase

/// @brief Stuct with all the member data of the LeafNode (useful during serialization of an openvdb LeafNode)
///
/// @note No client code should (or can) interface with this struct so it can safely be ignored!
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp4, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp4;
    using ArrayType = uint8_t;// type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint8_t mCode[1u << (3 * LOG2DIM - 1)];

    __hostdev__ static constexpr uint8_t bitWidth() { return 4u; }
    __hostdev__ float getValue(uint32_t i) const
    {
#if 0
        const uint8_t c = mCode[i>>1];
        return ( (i&1) ? c >> 4 : c & uint8_t(15) )*BaseT::mQuantum + BaseT::mMinimum;
#else
        return ((mCode[i>>1] >> ((i&1)<<2)) & uint8_t(15))*BaseT::mQuantum + BaseT::mMinimum;
#endif
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp4>

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp8, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp8;
    using ArrayType = uint8_t;// type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint8_t mCode[1u << 3 * LOG2DIM];

    __hostdev__ static constexpr uint8_t bitWidth() { return 8u; }
    __hostdev__ float getValue(uint32_t i) const
    {
        return mCode[i]*BaseT::mQuantum + BaseT::mMinimum;// code * (max-min)/255 + min
    }
    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp8>

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<Fp16, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = Fp16;
    using ArrayType = uint16_t;// type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;
    alignas(32) uint16_t mCode[1u << 3 * LOG2DIM];

    __hostdev__ static constexpr uint8_t bitWidth() { return 16u; }
    __hostdev__ float getValue(uint32_t i) const
    {
        return mCode[i]*BaseT::mQuantum + BaseT::mMinimum;// code * (max-min)/65535 + min
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<Fp16>

template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<FpN, CoordT, MaskT, LOG2DIM>
    : public LeafFnBase<CoordT, MaskT, LOG2DIM>
{
    using BaseT = LeafFnBase<CoordT, MaskT, LOG2DIM>;
    using BuildType = FpN;
    static constexpr bool FIXED_SIZE = false;

    __hostdev__ uint8_t bitWidth() const { return 1 << (BaseT::mFlags >> 5); }// 4,8,16,32 = 2^(2,3,4,5)
    __hostdev__ size_t memUsage() const { return sizeof(*this) + this->bitWidth()*64; }
    __hostdev__ static size_t memUsage(uint32_t bitWidth) { return 96u + bitWidth*64; }
    __hostdev__ float getValue(uint32_t i) const
    {
#ifdef NANOVDB_FPN_BRANCHLESS// faster
        const int b = BaseT::mFlags >> 5;// b = 0, 1, 2, 3, 4 corresponding to 1, 2, 4, 8, 16 bits
#if 0// use LUT
        uint16_t code = reinterpret_cast<const uint16_t*>(this + 1)[i >> (4 - b)];
        const static uint8_t shift[5] = {15, 7, 3, 1, 0};
        const static uint16_t mask[5] = {1, 3, 15, 255, 65535};
        code >>= (i & shift[b]) << b;
        code  &= mask[b];
#else// no LUT
        uint32_t code = reinterpret_cast<const uint32_t*>(this + 1)[i >> (5 - b)];
        //code >>= (i & ((16 >> b) - 1)) << b;
        code >>= (i & ((32 >> b) - 1)) << b;
        code  &= (1 << (1 << b)) - 1;
#endif
#else// use branched version (slow)
        float code;
        auto *values = reinterpret_cast<const uint8_t*>(this+1);
        switch (BaseT::mFlags >> 5) {
            case 0u:// 1 bit float
                code = float((values[i>>3] >>  (i&7)    ) & uint8_t(1));
                break;
            case 1u:// 2 bits float
                code = float((values[i>>2] >> ((i&3)<<1)) & uint8_t(3));
                break;
            case 2u:// 4 bits float
                code = float((values[i>>1] >> ((i&1)<<2)) & uint8_t(15));
                break;
            case 3u:// 8 bits float
                code = float(values[i]);
                break;
            default:// 16 bits float
                code = float(reinterpret_cast<const uint16_t*>(values)[i]);
        }
#endif
        return float(code) * BaseT::mQuantum + BaseT::mMinimum;// code * (max-min)/UNITS + min
    }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<FpN>

// Partial template specialization of LeafData with bool
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<bool, CoordT, MaskT, LOG2DIM>
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = bool;
    using BuildType = bool;
    using FloatType = bool;// dummy value type
    using ArrayType = MaskT<LOG2DIM>;// type used for the internal mValue array
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B.
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.
    MaskT<LOG2DIM> mValues; // LOG2DIM(3): 64B.

    //__hostdev__ const ValueType* values() const { return nullptr; }
    __hostdev__ bool getValue(uint32_t i) const { return mValues.isOn(i); }
    __hostdev__ bool getMin() const { return false; }// dummy
    __hostdev__ bool getMax() const { return false; }// dummy
    __hostdev__ bool getAvg() const { return false; }// dummy
    __hostdev__ bool getDev() const { return false; }// dummy
    __hostdev__ void setValue(uint32_t offset, bool v)
    {
        mValueMask.setOn(offset);
        mValues.set(offset, v);
    }

    __hostdev__ void setMin(const bool&) {}// no-op
    __hostdev__ void setMax(const bool&) {}// no-op
    __hostdev__ void setAvg(const bool&) {}// no-op
    __hostdev__ void setDev(const bool&) {}// no-op

    template <typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<bool>

// Partial template specialization of LeafData with ValueMask
template<typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
struct NANOVDB_ALIGN(NANOVDB_DATA_ALIGNMENT) LeafData<ValueMask, CoordT, MaskT, LOG2DIM>
{
    static_assert(sizeof(CoordT) == sizeof(Coord), "Mismatching sizeof");
    static_assert(sizeof(MaskT<LOG2DIM>) == sizeof(Mask<LOG2DIM>), "Mismatching sizeof");
    using ValueType = bool;
    using BuildType = ValueMask;
    using FloatType = bool;// dummy value type
    using ArrayType = void;// type used for the internal mValue array - void means missing
    static constexpr bool FIXED_SIZE = true;

    CoordT         mBBoxMin; // 12B.
    uint8_t        mBBoxDif[3]; // 3B.
    uint8_t        mFlags; // 1B.
    MaskT<LOG2DIM> mValueMask; // LOG2DIM(3): 64B.

    //__hostdev__ const ValueType* values() const { return nullptr; }
    __hostdev__ bool getValue(uint32_t i) const { return mValueMask.isOn(i); }
    __hostdev__ bool getMin() const { return false; }// dummy
    __hostdev__ bool getMax() const { return false; }// dummy
    __hostdev__ bool getAvg() const { return false; }// dummy
    __hostdev__ bool getDev() const { return false; }// dummy
    __hostdev__ void setValue(uint32_t offset, bool)
    {
        mValueMask.setOn(offset);
    }

    __hostdev__ void setMin(const ValueType&) {}// no-op
    __hostdev__ void setMax(const ValueType&) {}// no-op
    __hostdev__ void setAvg(const FloatType&) {}// no-op
    __hostdev__ void setDev(const FloatType&) {}// no-op

    template <typename T>
    __hostdev__ void setOrigin(const T& ijk) { mBBoxMin = ijk; }

    /// @brief This class cannot be constructed or deleted
    LeafData() = delete;
    LeafData(const LeafData&) = delete;
    LeafData& operator=(const LeafData&) = delete;
    ~LeafData() = delete;
}; // LeafData<ValueMask>

/// @brief Leaf nodes of the VDB tree. (defaults to 8x8x8 = 512 voxels)
template<typename BuildT,
         typename CoordT = Coord,
         template<uint32_t> class MaskT = Mask,
         uint32_t Log2Dim = 3>
class LeafNode : private LeafData<BuildT, CoordT, MaskT, Log2Dim>
{
public:
    struct ChildNodeType
    {
        __hostdev__ static uint32_t dim() { return 1u; }
    }; // Voxel
    using LeafNodeType = LeafNode<BuildT, CoordT, MaskT, Log2Dim>;
    using DataType = LeafData<BuildT, CoordT, MaskT, Log2Dim>;
    using ValueType = typename DataType::ValueType;
    using FloatType = typename DataType::FloatType;
    using BuildType = typename DataType::BuildType;
    using CoordType = CoordT;
    static constexpr bool FIXED_SIZE = DataType::FIXED_SIZE;
    template<uint32_t LOG2>
    using MaskType = MaskT<LOG2>;

    static_assert(is_same<ValueType,typename BuildToValueMap<BuildType>::Type>::value, "Mismatching BuildType");
    static constexpr uint32_t LOG2DIM = Log2Dim;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL; // number of voxels along each axis of this node
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = (1u << LOG2DIM) - 1u; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node

    __hostdev__ DataType* data() { return reinterpret_cast<DataType*>(this); }

    __hostdev__ const DataType* data() const { return reinterpret_cast<const DataType*>(this); }

    /// @brief Return a const reference to the bit mask of active voxels in this leaf node
    __hostdev__ const MaskType<LOG2DIM>& valueMask() const { return DataType::mValueMask; }

    /// @brief Return a const reference to the minimum active value encoded in this leaf node
    __hostdev__ ValueType minimum() const { return this->getMin(); }

    /// @brief Return a const reference to the maximum active value encoded in this leaf node
    __hostdev__ ValueType maximum() const { return this->getMax(); }

    /// @brief Return a const reference to the average of all the active values encoded in this leaf node
    __hostdev__ FloatType average() const { return DataType::getAvg(); }

    /// @brief Return the variance of all the active values encoded in this leaf node
    __hostdev__ FloatType variance() const { return DataType::getDev()*DataType::getDev(); }

    /// @brief Return a const reference to the standard deviation of all the active values encoded in this leaf node
    __hostdev__ FloatType stdDeviation() const { return DataType::getDev(); }

    __hostdev__ uint8_t flags() const { return DataType::mFlags; }

    /// @brief Return the origin in index space of this leaf node
    __hostdev__ CoordT origin() const { return DataType::mBBoxMin & ~MASK; }

    __hostdev__ static CoordT OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return CoordT(n >> 2 * LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    /// @brief Converts (in place) a local index coordinate to a global index coordinate
    __hostdev__ void localToGlobalCoord(Coord& ijk) const { ijk += this->origin(); }

    __hostdev__ CoordT offsetToGlobalCoord(uint32_t n) const
    {
        return OffsetToLocalCoord(n) + this->origin();
    }

    /// @brief Return the dimension, in index space, of this leaf node (typically 8 as for openvdb leaf nodes!)
    __hostdev__ static uint32_t dim() { return 1u << LOG2DIM; }

    /// @brief Return the bounding box in index space of active values in this leaf node
    __hostdev__ BBox<CoordT> bbox() const
    {
        BBox<CoordT> bbox(DataType::mBBoxMin, DataType::mBBoxMin);
        if ( this->isActive() ) {
            bbox.max()[0] += DataType::mBBoxDif[0];
            bbox.max()[1] += DataType::mBBoxDif[1];
            bbox.max()[2] += DataType::mBBoxDif[2];
        } else {// very rare case
            bbox = BBox<CoordT>();// invalid
        }
        return bbox;
    }

    /// @brief Return the total number of voxels (e.g. values) encoded in this leaf node
    __hostdev__ static uint32_t voxelCount() { return 1u << (3 * LOG2DIM); }

    /// @brief return memory usage in bytes for the class
    __hostdev__ static uint64_t memUsage() { return sizeof(LeafNodeType); }

    /// @brief This class cannot be constructed or deleted
    LeafNode() = delete;
    LeafNode(const LeafNode&) = delete;
    LeafNode& operator=(const LeafNode&) = delete;
    ~LeafNode() = delete;

    /// @brief Return the voxel value at the given offset.
    __hostdev__ ValueType getValue(uint32_t offset) const { return  DataType::getValue(offset); }

    /// @brief Return the voxel value at the given coordinate.
    __hostdev__ ValueType getValue(const CoordT& ijk) const { return  DataType::getValue(CoordToOffset(ijk)); }

    /// @brief Sets the value at the specified location and activate its state.
    ///
    /// @note This is safe since it does not change the topology of the tree (unlike setValue methods on the other nodes)
    __hostdev__ void setValue(const CoordT& ijk, const ValueType& v) { DataType::setValue(CoordToOffset(ijk), v); }

    /// @brief Sets the value at the specified location but leaves its state unchanged.
    ///
    /// @note This is safe since it does not change the topology of the tree (unlike setValue methods on the other nodes)
    __hostdev__ void setValueOnly(uint32_t offset, const ValueType& v) { DataType::setValueOnly(offset, v); }
    __hostdev__ void setValueOnly(const CoordT& ijk, const ValueType& v) { DataType::setValueOnly(CoordToOffset(ijk), v); }

    /// @brief Return @c true if the voxel value at the given coordinate is active.
    __hostdev__ bool isActive(const CoordT& ijk) const { return DataType::mValueMask.isOn(CoordToOffset(ijk)); }
    __hostdev__ bool isActive(uint32_t n) const { return DataType::mValueMask.isOn(n); }

    /// @brief Return @c true if any of the voxel value are active in this leaf node.
    __hostdev__ bool isActive() const
    {
        NANOVDB_ASSERT( bool(DataType::mFlags & uint8_t(2)) != DataType::mValueMask.isOff() );
        return DataType::mFlags & uint8_t(2);
    }


    /// @brief Return @c true if the voxel value at the given coordinate is active and updates @c v with the value.
    __hostdev__ bool probeValue(const CoordT& ijk, ValueType& v) const
    {
        const uint32_t n = CoordToOffset(ijk);
        v =  DataType::getValue(n);
        return DataType::mValueMask.isOn(n);
    }

    __hostdev__ const LeafNode* probeLeaf(const CoordT&) const { return this; }

    /// @brief Return the linear offset corresponding to the given coordinate
    __hostdev__ static uint32_t CoordToOffset(const CoordT& ijk)
    {
    #if 0
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) + ((ijk[1] & MASK) << LOG2DIM) + (ijk[2] & MASK);
    #else
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) | ((ijk[1] & MASK) << LOG2DIM) | (ijk[2] & MASK);
    #endif
    }

    /// @brief Updates the local bounding box of active voxels in this node.
    ///
    /// @warning It assumes that the origin and value mask have already been set.
    ///
    /// @details This method is based on few (intrinsic) bit operations and hence is relatively fast.
    ///          However, it should only only be called of either the value mask has changed or if the
    ///          active bounding box is still undefined. e.g. during constrution of this node.
    __hostdev__ void updateBBox();

private:
    static_assert(sizeof(DataType) % NANOVDB_DATA_ALIGNMENT == 0, "sizeof(LeafData) is misaligned");
    //static_assert(offsetof(DataType, mValues) % 32 == 0, "LeafData::mValues is misaligned");

    template<typename, int, int, int>
    friend class ReadAccessor;

    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;

    /// @brief Private method to return a voxel value and update a (dummy) ReadAccessor
    template<typename AccT>
    __hostdev__ ValueType getValueAndCache(const CoordT& ijk, const AccT&) const { return this->getValue(ijk); }

    /// @brief Return the node information.
    template<typename AccT>
    __hostdev__ typename AccT::NodeInfo getNodeInfoAndCache(const CoordType& /*ijk*/, const AccT& /*acc*/) const {
        using NodeInfoT = typename AccT::NodeInfo;
        return NodeInfoT{LEVEL, this->dim(), this->minimum(), this->maximum(),
                         this->average(), this->stdDeviation(), this->bbox()[0], this->bbox()[1]};
    }

    template<typename AccT>
    __hostdev__ bool isActiveAndCache(const CoordT& ijk, const AccT&) const { return this->isActive(ijk); }

    template<typename AccT>
    __hostdev__ bool probeValueAndCache(const CoordT& ijk, ValueType& v, const AccT&) const { return this->probeValue(ijk, v); }

    template<typename AccT>
    __hostdev__ const LeafNode* probeLeafAndCache(const CoordT&, const AccT&) const { return this; }

    template<typename RayT, typename AccT>
    __hostdev__ uint32_t getDimAndCache(const CoordT&, const RayT& /*ray*/, const AccT&) const
    {
        if (DataType::mFlags & uint8_t(1))
            return this->dim(); // skip this node if first bit is set
        //if (!ray.intersects( this->bbox() )) return 1 << LOG2DIM;
        return ChildNodeType::dim();
    }

}; // LeafNode class

template<typename ValueT, typename CoordT, template<uint32_t> class MaskT, uint32_t LOG2DIM>
__hostdev__ inline void LeafNode<ValueT, CoordT, MaskT, LOG2DIM>::updateBBox()
{
    static_assert(LOG2DIM == 3, "LeafNode::updateBBox: only supports LOGDIM = 3!");
    if (!this->isActive()) return;
    auto update = [&](uint32_t min, uint32_t max, int axis) {
        NANOVDB_ASSERT(min <= max && max < 8);
        DataType::mBBoxMin[axis] = (DataType::mBBoxMin[axis] & ~MASK) + int(min);
        DataType::mBBoxDif[axis] = uint8_t(max - min);
    };
    uint64_t word64 = DataType::mValueMask.template getWord<uint64_t>(0);
    uint32_t Xmin = word64 ? 0u : 8u;
    uint32_t Xmax = Xmin;
    for (int i = 1; i < 8; ++i) { // last loop over 8 64 words
        if (uint64_t w = DataType::mValueMask.template getWord<uint64_t>(i)) { // skip if word has no set bits
            word64 |= w; // union 8 x 64 bits words into one 64 bit word
            if (Xmin == 8) {
                Xmin = i; // only set once
            }
            Xmax = i;
        }
    }
    NANOVDB_ASSERT(word64);
    update(Xmin, Xmax, 0);
    update(FindLowestOn(word64) >> 3, FindHighestOn(word64) >> 3, 1);
    const uint32_t *p = reinterpret_cast<const uint32_t*>(&word64), word32 = p[0] | p[1];
    const uint16_t *q = reinterpret_cast<const uint16_t*>(&word32), word16 = q[0] | q[1];
    const uint8_t  *b = reinterpret_cast<const uint8_t* >(&word16), byte   = b[0] | b[1];
    NANOVDB_ASSERT(byte);
    update(FindLowestOn(static_cast<uint32_t>(byte)), FindHighestOn(static_cast<uint32_t>(byte)), 2);
} // LeafNode::updateBBox

// --------------------------> Template specializations and traits <------------------------------------

/// @brief Template specializations to the default configuration used in OpenVDB:
///        Root -> 32^3 -> 16^3 -> 8^3
template<typename BuildT>
using NanoLeaf = LeafNode<BuildT, Coord, Mask, 3>;
template<typename BuildT>
using NanoLower = InternalNode<NanoLeaf<BuildT>, 4>;
template<typename BuildT>
using NanoUpper = InternalNode<NanoLower<BuildT>, 5>;
template<typename BuildT>
using NanoRoot = RootNode<NanoUpper<BuildT>>;
template<typename BuildT>
using NanoTree = Tree<NanoRoot<BuildT>>;
template<typename BuildT>
using NanoGrid = Grid<NanoTree<BuildT>>;

/// @brief Trait to map from LEVEL to node type
template<typename BuildT, int LEVEL>
struct NanoNode;

// Partial template specialization of above Node struct
template<typename BuildT>
struct NanoNode<BuildT, 0>
{
    using Type = NanoLeaf<BuildT>;
    using type = NanoLeaf<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 1>
{
    using Type = NanoLower<BuildT>;
    using type = NanoLower<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 2>
{
    using Type = NanoUpper<BuildT>;
    using type = NanoUpper<BuildT>;
};
template<typename BuildT>
struct NanoNode<BuildT, 3>
{
    using Type = NanoRoot<BuildT>;
    using type = NanoRoot<BuildT>;
};

using FloatTree  = NanoTree<float>;
using DoubleTree = NanoTree<double>;
using Int32Tree  = NanoTree<int32_t>;
using UInt32Tree = NanoTree<uint32_t>;
using Int64Tree  = NanoTree<int64_t>;
using Vec3fTree  = NanoTree<Vec3f>;
using Vec3dTree  = NanoTree<Vec3d>;
using Vec4fTree  = NanoTree<Vec4f>;
using Vec4dTree  = NanoTree<Vec4d>;
using Vec3ITree  = NanoTree<Vec3i>;
using MaskTree   = NanoTree<ValueMask>;
using BoolTree   = NanoTree<bool>;

using FloatGrid  = Grid<FloatTree>;
using DoubleGrid = Grid<DoubleTree>;
using Int32Grid  = Grid<Int32Tree>;
using UInt32Grid = Grid<UInt32Tree>;
using Int64Grid  = Grid<Int64Tree>;
using Vec3fGrid  = Grid<Vec3fTree>;
using Vec3dGrid  = Grid<Vec3dTree>;
using Vec4fGrid  = Grid<Vec4fTree>;
using Vec4dGrid  = Grid<Vec4dTree>;
using Vec3IGrid  = Grid<Vec3ITree>;
using MaskGrid   = Grid<MaskTree>;
using BoolGrid   = Grid<BoolTree>;

// --------------------------> ReadAccessor <------------------------------------

/// @brief A read-only value accessor with three levels of node caching. This allows for
///        inverse tree traversal during lookup, which is on average significantly faster
///        than calling the equivalent method on the tree (i.e. top-down traversal).
///
/// @note  By virtue of the fact that a value accessor accelerates random access operations
///        by re-using cached access patterns, this access should be reused for multiple access
///        operations. In other words, never create an instance of this accessor for a single
///        acccess only. In general avoid single access operations with this accessor, and
///        if that is not possible call the corresponding method on the tree instead.
///
/// @warning Since this ReadAccessor internally caches raw pointers to the nodes of the tree
///          structure, it is not safe to copy between host and device, or even to share among
///          multiple threads on the same host or device. However, it is light-weight so simple
///          instantiate one per thread (on the host and/or device).
///
/// @details Used to accelerated random access into a VDB tree. Provides on average
///          O(1) random access operations by means of inverse tree traversal,
///          which amortizes the non-const time complexity of the root node.

template <typename BuildT>
class ReadAccessor<BuildT, -1, -1, -1>
{
    using RootT  = NanoRoot<BuildT>; // root node
    using LeafT  = NanoLeaf<BuildT>; // Leaf node
    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordType::ValueType;

    mutable const RootT* mRoot; // 8 bytes (mutable to allow for access methods to be const)
public:
    using ValueType = typename RootT::ValueType;
    using CoordType = typename RootT::CoordType;

    static const int CacheLevels = 0;

    struct NodeInfo {
        uint32_t  mLevel; //   4B
        uint32_t  mDim; //     4B
        ValueType mMinimum; // typically 4B
        ValueType mMaximum; // typically 4B
        FloatType mAverage; // typically 4B
        FloatType mStdDevi; // typically 4B
        CoordType mBBoxMin; // 3*4B
        CoordType mBBoxMax; // 3*4B
    };

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root) : mRoot{&root} {}

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        return mRoot->getValueAndCache(ijk, *this);
    }

    __hostdev__ NodeInfo getNodeInfo(const CoordType& ijk) const
    {
        return mRoot->getNodeInfoAndCache(ijk, *this);
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        return mRoot->isActiveAndCache(ijk, *this);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        return mRoot->probeValueAndCache(ijk, v, *this);
    }

    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const
    {
        return mRoot->probeLeafAndCache(ijk, *this);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief No-op
    template<typename NodeT>
    __hostdev__ void insert(const CoordType&, const NodeT*) const {}
}; // ReadAccessor<ValueT, -1, -1, -1> class

/// @brief Node caching at a single tree level
template <typename BuildT, int LEVEL0>
class ReadAccessor<BuildT, LEVEL0, -1, -1>//e.g. 0, 1, 2
{
    static_assert(LEVEL0 >= 0 && LEVEL0 <= 2, "LEVEL0 should be 0, 1, or 2");

    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>; //  root node
    using LeafT  = NanoLeaf<BuildT>; // Leaf node
    using NodeT  = typename NodeTrait<TreeT, LEVEL0>::type;
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;

    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
    mutable CoordT       mKey; // 3*4 = 12 bytes
    mutable const RootT* mRoot; // 8 bytes
    mutable const NodeT* mNode; // 8 bytes

public:
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 1;

    using NodeInfo = typename ReadAccessor<ValueT, -1, -1, -1>::NodeInfo;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
        : mKey(CoordType::max())
        , mRoot(&root)
        , mNode(nullptr)
    {
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    __hostdev__ bool isCached(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKey[0] &&
               (ijk[1] & int32_t(~NodeT::MASK)) == mKey[1] &&
               (ijk[2] & int32_t(~NodeT::MASK)) == mKey[2];
    }

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
        if (this->isCached(ijk)) {
            return mNode->getValueAndCache(ijk, *this);
        }
        return mRoot->getValueAndCache(ijk, *this);
    }

    __hostdev__ NodeInfo getNodeInfo(const CoordType& ijk) const
    {
        if (this->isCached(ijk)) {
            return mNode->getNodeInfoAndCache(ijk, *this);
        }
        return mRoot->getNodeInfoAndCache(ijk, *this);
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
        if (this->isCached(ijk)) {
            return mNode->isActiveAndCache(ijk, *this);
        }
        return mRoot->isActiveAndCache(ijk, *this);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
        if (this->isCached(ijk)) {
            return mNode->probeValueAndCache(ijk, v, *this);
        }
        return mRoot->probeValueAndCache(ijk, v, *this);
    }

    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const
    {
        if (this->isCached(ijk)) {
            return mNode->probeLeafAndCache(ijk, *this);
        }
        return mRoot->probeLeafAndCache(ijk, *this);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
        if (this->isCached(ijk)) {
            return mNode->getDimAndCache(ijk, ray, *this);
        }
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    __hostdev__ void insert(const CoordType& ijk, const NodeT* node) const
    {
        mKey = ijk & ~NodeT::MASK;
        mNode = node;
    }

    // no-op
    template<typename OtherNodeT>
    __hostdev__ void insert(const CoordType&, const OtherNodeT*) const {}

}; // ReadAccessor<ValueT, LEVEL0>

template <typename BuildT, int LEVEL0, int LEVEL1>
class ReadAccessor<BuildT, LEVEL0, LEVEL1, -1>//e.g. (0,1), (1,2), (0,2)
{
    static_assert(LEVEL0 >= 0 && LEVEL0 <= 2, "LEVEL0 must be 0, 1, 2");
    static_assert(LEVEL1 >= 0 && LEVEL1 <= 2, "LEVEL1 must be 0, 1, 2");
    static_assert(LEVEL0 < LEVEL1, "Level 0 must be lower than level 1");
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    using Node1T = typename NodeTrait<TreeT, LEVEL0>::type;
    using Node2T = typename NodeTrait<TreeT, LEVEL1>::type;
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;
    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
#ifdef USE_SINGLE_ACCESSOR_KEY // 44 bytes total
    mutable CoordT mKey; // 3*4 = 12 bytes
#else // 68 bytes total
    mutable CoordT mKeys[2]; // 2*3*4 = 24 bytes
#endif
    mutable const RootT*  mRoot;
    mutable const Node1T* mNode1;
    mutable const Node2T* mNode2;

public:
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 2;

    using NodeInfo = typename ReadAccessor<ValueT,-1,-1,-1>::NodeInfo;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
#ifdef USE_SINGLE_ACCESSOR_KEY
        : mKey(CoordType::max())
#else
        : mKeys{CoordType::max(), CoordType::max()}
#endif
        , mRoot(&root)
        , mNode1(nullptr)
        , mNode2(nullptr)
    {
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

#ifdef USE_SINGLE_ACCESSOR_KEY
    __hostdev__ bool isCached1(CoordValueType dirty) const
    {
        if (!mNode1)
            return false;
        if (dirty & int32_t(~Node1T::MASK)) {
            mNode1 = nullptr;
            return false;
        }
        return true;
    }
    __hostdev__ bool isCached2(CoordValueType dirty) const
    {
        if (!mNode2)
            return false;
        if (dirty & int32_t(~Node2T::MASK)) {
            mNode2 = nullptr;
            return false;
        }
        return true;
    }
    __hostdev__ CoordValueType computeDirty(const CoordType& ijk) const
    {
        return (ijk[0] ^ mKey[0]) | (ijk[1] ^ mKey[1]) | (ijk[2] ^ mKey[2]);
    }
#else
    __hostdev__ bool isCached1(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~Node1T::MASK)) == mKeys[0][0] &&
               (ijk[1] & int32_t(~Node1T::MASK)) == mKeys[0][1] &&
               (ijk[2] & int32_t(~Node1T::MASK)) == mKeys[0][2];
    }
    __hostdev__ bool isCached2(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~Node2T::MASK)) == mKeys[1][0] &&
               (ijk[1] & int32_t(~Node2T::MASK)) == mKeys[1][1] &&
               (ijk[2] & int32_t(~Node2T::MASK)) == mKeys[1][2];
    }
#endif

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->getValueAndCache(ijk, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->getValueAndCache(ijk, *this);
        }
        return mRoot->getValueAndCache(ijk, *this);
    }

    __hostdev__ NodeInfo getNodeInfo(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->getNodeInfoAndCache(ijk, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->getNodeInfoAndCache(ijk, *this);
        }
        return mRoot->getNodeInfoAndCache(ijk, *this);
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->isActiveAndCache(ijk, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->isActiveAndCache(ijk, *this);
        }
        return mRoot->isActiveAndCache(ijk, *this);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->probeValueAndCache(ijk, v, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->probeValueAndCache(ijk, v, *this);
        }
        return mRoot->probeValueAndCache(ijk, v, *this);
    }

    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->probeLeafAndCache(ijk, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->probeLeafAndCache(ijk, *this);
        }
        return mRoot->probeLeafAndCache(ijk, *this);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached1(dirty)) {
            return mNode1->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached2(dirty)) {
            return mNode2->getDimAndCache(ijk, ray, *this);
        }
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    __hostdev__ void insert(const CoordType& ijk, const Node1T* node) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[0] = ijk & ~Node1T::MASK;
#endif
        mNode1 = node;
    }
    __hostdev__ void insert(const CoordType& ijk, const Node2T* node) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[1] = ijk & ~Node2T::MASK;
#endif
        mNode2 = node;
    }
    template <typename OtherNodeT>
    __hostdev__ void insert(const CoordType&, const OtherNodeT*) const {}
}; // ReadAccessor<BuildT, LEVEL0, LEVEL1>


/// @brief Node caching at all (three) tree levels
template <typename BuildT>
class ReadAccessor<BuildT, 0, 1, 2>
{
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>; //  root node
    using NodeT2 = NanoUpper<BuildT>; // upper internal node
    using NodeT1 = NanoLower<BuildT>; // lower internal node
    using LeafT  = NanoLeaf< BuildT>; // Leaf node
    using CoordT = typename RootT::CoordType;
    using ValueT = typename RootT::ValueType;

    using FloatType = typename RootT::FloatType;
    using CoordValueType = typename RootT::CoordT::ValueType;

    // All member data are mutable to allow for access methods to be const
#ifdef USE_SINGLE_ACCESSOR_KEY // 44 bytes total
    mutable CoordT mKey; // 3*4 = 12 bytes
#else // 68 bytes total
    mutable CoordT mKeys[3]; // 3*3*4 = 36 bytes
#endif
    mutable const RootT* mRoot;
    mutable const void* mNode[3]; // 4*8 = 32 bytes

public:
    using ValueType = ValueT;
    using CoordType = CoordT;

    static const int CacheLevels = 3;

    using NodeInfo = typename ReadAccessor<ValueT, -1, -1, -1>::NodeInfo;

    /// @brief Constructor from a root node
    __hostdev__ ReadAccessor(const RootT& root)
#ifdef USE_SINGLE_ACCESSOR_KEY
        : mKey(CoordType::max())
#else
        : mKeys{CoordType::max(), CoordType::max(), CoordType::max()}
#endif
        , mRoot(&root)
        , mNode{nullptr, nullptr, nullptr}
    {
    }

    __hostdev__ const RootT& root() const { return *mRoot; }

    /// @brief Defaults constructors
    ReadAccessor(const ReadAccessor&) = default;
    ~ReadAccessor() = default;
    ReadAccessor& operator=(const ReadAccessor&) = default;

    /// @brief Return a const point to the cached node of the specified type
    ///
    /// @warning The return value could be NULL.
    template<typename NodeT>
    __hostdev__ const NodeT* getNode() const
    {
        using T = typename NodeTrait<TreeT, NodeT::LEVEL>::type;
        static_assert(is_same<T, NodeT>::value, "ReadAccessor::getNode: Invalid node type");
        return reinterpret_cast<const T*>(mNode[NodeT::LEVEL]);
    }

#ifdef USE_SINGLE_ACCESSOR_KEY
    template<typename NodeT>
    __hostdev__ bool isCached(CoordValueType dirty) const
    {
        if (!mNode[NodeT::LEVEL])
            return false;
        if (dirty & int32_t(~NodeT::MASK)) {
            mNode[NodeT::LEVEL] = nullptr;
            return false;
        }
        return true;
    }

    __hostdev__ CoordValueType computeDirty(const CoordType& ijk) const
    {
        return (ijk[0] ^ mKey[0]) | (ijk[1] ^ mKey[1]) | (ijk[2] ^ mKey[2]);
    }
#else
    template<typename NodeT>
    __hostdev__ bool isCached(const CoordType& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][0] && (ijk[1] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][1] && (ijk[2] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][2];
    }
#endif

    __hostdev__ ValueType getValue(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->getValue(ijk);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getValueAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getValueAndCache(ijk, *this);
        }
        return mRoot->getValueAndCache(ijk, *this);
    }

    __hostdev__ NodeInfo getNodeInfo(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->getNodeInfoAndCache(ijk, *this);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getNodeInfoAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getNodeInfoAndCache(ijk, *this);
        }
        return mRoot->getNodeInfoAndCache(ijk, *this);
    }

    __hostdev__ bool isActive(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->isActive(ijk);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->isActiveAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->isActiveAndCache(ijk, *this);
        }
        return mRoot->isActiveAndCache(ijk, *this);
    }

    __hostdev__ bool probeValue(const CoordType& ijk, ValueType& v) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->probeValue(ijk, v);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->probeValueAndCache(ijk, v, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->probeValueAndCache(ijk, v, *this);
        }
        return mRoot->probeValueAndCache(ijk, v, *this);
    }

    __hostdev__ const LeafT* probeLeaf(const CoordType& ijk) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0]);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->probeLeafAndCache(ijk, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->probeLeafAndCache(ijk, *this);
        }
        return mRoot->probeLeafAndCache(ijk, *this);
    }

    template<typename RayT>
    __hostdev__ uint32_t getDim(const CoordType& ijk, const RayT& ray) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        const CoordValueType dirty = this->computeDirty(ijk);
#else
        auto&& dirty = ijk;
#endif
        if (this->isCached<LeafT>(dirty)) {
            return ((LeafT*)mNode[0])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT1>(dirty)) {
            return ((NodeT1*)mNode[1])->getDimAndCache(ijk, ray, *this);
        } else if (this->isCached<NodeT2>(dirty)) {
            return ((NodeT2*)mNode[2])->getDimAndCache(ijk, ray, *this);
        }
        return mRoot->getDimAndCache(ijk, ray, *this);
    }

private:
    /// @brief Allow nodes to insert themselves into the cache.
    template<typename>
    friend class RootNode;
    template<typename, uint32_t>
    friend class InternalNode;
    template<typename, typename, template<uint32_t> class, uint32_t>
    friend class LeafNode;

    /// @brief Inserts a leaf node and key pair into this ReadAccessor
    template<typename NodeT>
    __hostdev__ void insert(const CoordType& ijk, const NodeT* node) const
    {
#ifdef USE_SINGLE_ACCESSOR_KEY
        mKey = ijk;
#else
        mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
#endif
        mNode[NodeT::LEVEL] = node;
    }
}; // ReadAccessor<BuildT, 0, 1, 2>

//////////////////////////////////////////////////

/// @brief Free-standing function for convenient creation of a ReadAccessor with
///        optional and customizable node caching.
///
/// @details createAccessor<>(grid):  No caching of nodes and hence it's thread-safe but slow
///          createAccessor<0>(grid): Caching of leaf nodes only
///          createAccessor<1>(grid): Caching of lower internal nodes only
///          createAccessor<2>(grid): Caching of upper internal nodes only
///          createAccessor<0,1>(grid): Caching of leaf and lower internal nodes
///          createAccessor<0,2>(grid): Caching of leaf and upper internal nodes
///          createAccessor<1,2>(grid): Caching of lower and upper internal nodes
///          createAccessor<0,1,2>(grid): Caching of all nodes at all tree levels

template <int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoGrid<ValueT> &grid)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(grid.tree().root());
}

template <int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoTree<ValueT> &tree)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(tree().root());
}

template <int LEVEL0 = -1, int LEVEL1 = -1, int LEVEL2 = -1, typename ValueT = float>
ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2> createAccessor(const NanoRoot<ValueT> &root)
{
    return ReadAccessor<ValueT, LEVEL0, LEVEL1, LEVEL2>(root);
}

//////////////////////////////////////////////////

/// @brief This is a convenient class that allows for access to grid meta-data
///        that are independent of the value type of a grid. That is, this class
///        can be used to get information about a grid without actually knowing
///        its ValueType.
class GridMetaData
{
    // We cast to a grid templated on a dummy ValueType which is safe because we are very
    // careful only to call certain methods which are known to be invariant to the ValueType!
    // In other words, don't use this technique unless you are intimately familiar with the
    // memory-layout of the data structure and the reasons why certain methods are safe
    // to call and others are not!
    using GridT = NanoGrid<int>;
    __hostdev__ const GridT& grid() const { return *reinterpret_cast<const GridT*>(this); }

public:
    __hostdev__ bool        isValid() const { return this->grid().isValid(); }
    __hostdev__ uint64_t    gridSize() const { return this->grid().gridSize(); }
    __hostdev__ uint32_t    gridIndex() const { return this->grid().gridIndex(); }
    __hostdev__ uint32_t    gridCount() const { return this->grid().gridCount(); }
    __hostdev__ const char* shortGridName() const { return this->grid().shortGridName(); }
    __hostdev__ GridType    gridType() const { return this->grid().gridType(); }
    __hostdev__ GridClass   gridClass() const { return this->grid().gridClass(); }
    __hostdev__ bool        isLevelSet() const { return this->grid().isLevelSet(); }
    __hostdev__ bool        isFogVolume() const { return this->grid().isFogVolume(); }
    __hostdev__ bool        isPointIndex() const { return this->grid().isPointIndex(); }
    __hostdev__ bool        isPointData() const { return this->grid().isPointData(); }
    __hostdev__ bool        isMask() const { return this->grid().isMask(); }
    __hostdev__ bool        isStaggered() const { return this->grid().isStaggered(); }
    __hostdev__ bool        isUnknown() const { return this->grid().isUnknown(); }
    __hostdev__ const Map& map() const { return this->grid().map(); }
    __hostdev__ const BBox<Vec3R>& worldBBox() const { return this->grid().worldBBox(); }
    __hostdev__ const BBox<Coord>& indexBBox() const { return this->grid().indexBBox(); }
    __hostdev__ Vec3R              voxelSize() const { return this->grid().voxelSize(); }
    __hostdev__ int                blindDataCount() const { return this->grid().blindDataCount(); }
    __hostdev__ const GridBlindMetaData& blindMetaData(int n) const { return this->grid().blindMetaData(n); }
    __hostdev__ uint64_t                 activeVoxelCount() const { return this->grid().activeVoxelCount(); }
    __hostdev__ uint32_t                 activeTileCount(uint32_t n) const { return this->grid().tree().activeTileCount(n); }
    __hostdev__ uint32_t                 nodeCount(uint32_t level) const { return this->grid().tree().nodeCount(level); }
    __hostdev__ uint64_t                 checksum() const { return this->grid().checksum(); }
    __hostdev__ bool                     isEmpty() const { return this->grid().isEmpty(); }
    __hostdev__ Version                  version() const { return this->grid().version(); }
}; // GridMetaData

/// @brief Class to access points at a specific voxel location
template<typename AttT>
class PointAccessor : public DefaultReadAccessor<uint32_t>
{
    using AccT = DefaultReadAccessor<uint32_t>;
    const UInt32Grid* mGrid;
    const AttT*       mData;

public:
    using LeafNodeType = typename NanoRoot<uint32_t>::LeafNodeType;

    PointAccessor(const UInt32Grid& grid)
        : AccT(grid.tree().root())
        , mGrid(&grid)
        , mData(reinterpret_cast<const AttT*>(grid.blindData(0)))
    {
        NANOVDB_ASSERT(grid.gridType() == GridType::UInt32);
        NANOVDB_ASSERT((grid.gridClass() == GridClass::PointIndex && is_same<uint32_t, AttT>::value) ||
                       (grid.gridClass() == GridClass::PointData && is_same<Vec3f, AttT>::value));
        NANOVDB_ASSERT(grid.blindDataCount() >= 1);
    }
    /// @brief Return the total number of point in the grid and set the
    ///        iterators to the complete range of points.
    __hostdev__ uint64_t gridPoints(const AttT*& begin, const AttT*& end) const
    {
        const uint64_t count = mGrid->blindMetaData(0).mElementCount;
        begin = mData;
        end = begin + count;
        return count;
    }
    /// @brief Return the number of points in the leaf node containing the coordinate @a ijk.
    ///        If this return value is larger than zero then the iterators @a begin and @a end
    ///        will point to all the attributes contained within that leaf node.
    __hostdev__ uint64_t leafPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr) {
            return 0;
        }
        begin = mData + leaf->minimum();
        end = begin + leaf->maximum();
        return leaf->maximum();
    }

    /// @brief get iterators over offsets to points at a specific voxel location
    __hostdev__ uint64_t voxelPoints(const Coord& ijk, const AttT*& begin, const AttT*& end) const
    {
        auto* leaf = this->probeLeaf(ijk);
        if (leaf == nullptr)
            return 0;
        const uint32_t offset = LeafNodeType::CoordToOffset(ijk);
        if (leaf->isActive(offset)) {
            auto* p = mData + leaf->minimum();
            begin = p + (offset == 0 ? 0 : leaf->getValue(offset - 1));
            end = p + leaf->getValue(offset);
            return end - begin;
        }
        return 0;
    }
}; // PointAccessor

} // namespace nanovdb

#endif // end of NANOVDB_NANOVDB_H_HAS_BEEN_INCLUDED

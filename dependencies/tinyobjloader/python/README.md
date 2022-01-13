# tinyobjloader, Wavefront .obj loader

`tinyobjloader` is a python wrapper for C++ wavefront .obj loader.
`tinyobjloader` is rather fast and feature rich than other pure python version of .obj loader.

## Requirements

* python 3.x(3.6+ recommended)
  * python 2.7 may work, but not officially supported.

## Install

You can install `tinyobjloader` with pip.

```
$ pip install tinyobjloader
```

## Quick tutorial

```py
import sys
import tinyobjloader

# Create reader.
reader = tinyobjloader.ObjReader()

filename = "cornellbox.obj"

# Load .obj(and .mtl) using default configuration
ret = reader.ParseFromFile(filename)

if ret == False:
    print("Warn:", reader.Warning())
    pint("Err:", reader.Error())
    print("Failed to load : ", filename)

    sys.exit(-1)

if reader.Warning():
    print("Warn:", reader.Warning())

attrib = reader.GetAttrib()
print("attrib.vertices = ", len(attrib.vertices))
print("attrib.normals = ", len(attrib.normals))
print("attrib.texcoords = ", len(attrib.texcoords))

materials = reader.GetMaterials()
print("Num materials: ", len(materials))
for m in materials:
    print(m.name)
    print(m.diffuse)

shapes = reader.GetShapes()
print("Num shapes: ", len(shapes))
for shape in shapes:
    print(shape.name)
    print("num_indices = {}".format(len(shape.mesh.indices)))

```

## More detailed usage

Please take a look at `python/sample.py` file in tinyobjloader git repo.

https://github.com/syoyo/tinyobjloader/blob/master/python/sample.py

## How to build

Using `cibuildwheel` is an recommended way to build a python module.
See $tinyobjloader/azure-pipelines.yml for details.

### Developer build

Edit `setup.py` and uncomment `Developer option` lines

Assume pip is installed.

```
$ pip install pybind11
$ python setup.py build
```

## License

MIT(tinyobjloader) and ISC(mapbox earcut) license.

## TODO
 * [ ] Writer saver


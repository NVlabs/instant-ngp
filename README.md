# Instant Neural Graphics Primitives ![](https://github.com/NVlabs/instant-ngp/workflows/CI/badge.svg)

<img src="docs/assets_readme/fox.gif" height="342"/> <img src="docs/assets_readme/robot5.gif" height="342"/>

Ever wanted to train a NeRF model of a fox in under 5 seconds? Or fly around a scene captured from photos of a factory robot? Of course you have!

Here you will find an implementation of four __neural graphics primitives__, being neural radiance fields (NeRF), signed distance functions (SDFs), neural images, and neural volumes.
In each case, we train and render a MLP with multiresolution hash input encoding using the [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) framework.

> __Instant Neural Graphics Primitives with a Multiresolution Hash Encoding__  
> [Thomas Müller](https://tom94.net), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _arXiv [cs.GR], Jan 2022_  
> __[ [Project page](https://nvlabs.github.io/instant-ngp) ] [ [Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf) ] [ [Video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4) ]__

 
For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)


## Requirements

- Both Windows and Linux are supported.
- An __NVIDIA GPU__ with __compute capability 75+ (RTX 20XX or newer)__. All shown results come from an RTX 3090.
- __[CUDA](https://developer.nvidia.com/cuda-toolkit) v10.2 or higher__, a __C++14__ capable compiler, and __[CMake](https://cmake.org/) v3.19 or higher__.
- __(optional) [Python](https://www.python.org/) 3.7 or higher__ for interactive bindings. Also, run `pip install -r requirements.txt`.
  - On some machines, `pyexr` refuses to install via `pip`. This can be resolved by installing OpenEXR from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr).
- __(optional) [OptiX](https://developer.nvidia.com/optix) 7.3 or higher__ for faster mesh SDF training. Set the environment variable `OptiX_INSTALL_DIR` to the installation directory if it is not discovered automatically.


If you are using Linux, install the following packages
```sh
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
```

We also recommend installing [CUDA](https://developer.nvidia.com/cuda-toolkit) and [OptiX](https://developer.nvidia.com/optix) in `/usr/local/` and adding the CUDA installation to your path.
For example, if you have CUDA 11.4, add the following to your `~/.bashrc`
```sh
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```


## Compilation (Windows & Linux)

Begin by cloning this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/nvlabs/instant-ngp
$ cd instant-ngp
```

Then, use CMake to build the project:
```sh
instant-ngp$ cmake . -B build
instant-ngp$ cmake --build build --config RelWithDebInfo -j 16
```

If the build succeeded, you can now run the code via the `build/testbed` executable or the `scripts/run.py` script described below.

If automatic GPU architecture detection fails, (as can happen if you have multiple GPUs installed), set the  `TCNN_CUDA_ARCHITECTURES` enivonment variable for the GPU you would like to use. Set it to `86` for RTX 3000 cards, `80` for A100 cards, and  `75` for RTX 2000 cards.


## Interactive training and rendering

<img src="docs/assets_readme/testbed.png" width="100%"/>

This codebase comes with an interactive testbed that includes many features beyond our academic publication:
- Additional training features, such as extrinsics and intrinsics optimization.
- Marching cubes for `NeRF->Mesh` and `SDF->Mesh` conversion.
- A spline-based camera path editor to create videos.
- Debug visualizations of the activations of every neuron input and output.
- And many more task-specific settings.
- See also our [one minute demonstration video of the tool](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4).



### NeRF fox

One test scene is provided in this repository, using a small number of frames from a casually captured phone video:

```sh
instant-ngp$ ./build/testbed --scene data/nerf/fox
```

<img src="docs/assets_readme/fox.png"/>

Alternatively, download any NeRF-compatible scene (e.g. [from the NeRF authors' drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)).
Now you can run:

```sh
instant-ngp$ ./build/testbed --scene data/nerf_synthetic/lego
```


### SDF armadillo

```sh
instant-ngp$ ./build/testbed --scene data/sdf/armadillo.obj
```

<img src="docs/assets_readme/armadillo.png"/>

### Image of Einstein

```sh
instant-ngp$ ./build/testbed --scene data/image/albert.exr
```

<img src="docs/assets_readme/albert.png"/>

To reproduce the gigapixel results, download, for example, [the Tokyo image](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837) and convert it to `.bin` using the `scripts/image2bin.py` script. This custom format improves compatibility and loading speed when resolution is high. Now you can run:

```sh
instant-ngp$ ./build/testbed --scene data/image/tokyo.bin
```


### Volume Renderer

Download the [nanovdb volume for the Disney cloud](https://drive.google.com/drive/folders/1SuycSAOSG64k2KLV7oWgyNWyCvZAkafK?usp=sharing), which is derived [from here](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/) ([CC BY-SA 3.0](https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf)).

```sh
instant-ngp$ ./build/testbed --mode volume --scene data/volume/wdas_cloud_quarter.nvdb
```
<img src="docs/assets_readme/cloud.png"/>

## Preparing new NeRF datasets

Our NeRF implementation expects initial camera parameters to be provided in a `transforms.json` file in a format compatible with [the original NeRF codebase](https://www.matthewtancik.com/nerf).
We provide a script as a convenience, `scripts/colmap2nerf.py`, that can be used to process a video file or sequence of images, using the open source [COLMAP](https://colmap.github.io/) structure from motion software to extract the necessary camera data.

Make sure that you have installed [COLMAP](https://colmap.github.io/) and that it is available in your PATH. If you are using a video file as input, also be sure to install [FFMPEG](https://www.ffmpeg.org/) and make sure that it is available in your PATH.
To check that this is the case, from a terminal window, you should be able to run `colmap` and `ffmpeg -?` and see some help text from each.

If you are training from a video file, run the `colmap2nerf.py` script from the folder containing the video, with the following recommended parameters:

```sh
data-folder$ python [path-to-instant-ngp]/scripts/colmap2nerf.py --video_in <filename of video> --video_fps 2 --run_colmap --aabb_scale 16
```

The above assumes a single video file as input, which then has frames extracted at the specified framerate (2). It is recommended to choose a frame rate that leads to around 50-150 images. So for a one minute video, `--video_fps 2` is ideal.

For training from images, place them in a subfolder called `images` and then use suitable options such as the ones below:

```sh
data-folder$ python [path-to-instant-ngp]/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16
```

The script will run ffmpeg and/or COLMAP as needed, followed by a conversion step to the required `transforms.json` format, which will be written in the current directory. 

By default, the script invokes colmap with the 'sequential matcher', which is suitable for images taken from a smoothly changing camera path, as in a video. The exhaustive matcher is more appropriate if the images are in no particular order, as shown in the image example above.
For more options, you can run the script with `--help`. For more advanced uses of COLMAP or for challenging scenes, please see the [COLMAP documentation](https://colmap.github.io/cli.html); you may need to modify the `scripts/colmap2nerf.py` script itself.

The `aabb_scale` parameter is the most important `instant-ngp` specific parameter. It specifies the extent of the scene, defaulting to 1; that is, the scene is scaled such that the camera positions are at an average distance of 1 unit from the origin. For small synthetic scenes such as the original NeRF dataset, the default `aabb_scale` of 1 is ideal and leads to fastest training. The NeRF model makes the assumption that the training images can entirely be explained by a scene contained within this bounding box. However, for natural scenes where there is a background that extends beyond this bounding box, the NeRF model will struggle and may hallucinate 'floaters' at the boundaries of the box. By setting `aabb_scale` to a larger power of 2 (up to a maximum of 16), the NeRF model will extend rays to a much larger bounding box. Note that this can impact training speed slightly. If in doubt, for natural scenes, start with an `aabb_scale` of 16, and subsequently reduce it if possible. The value can be directly edited in the `transforms.json` output file, without re-running the `colmap2nerf` script.

Assuming success, you can now train your NeRF model as follows, starting in the `instant-ngp` folder:

```sh
instant-ngp$ ./build/testbed --mode nerf --scene [path to training data folder containing transforms.json]
```

### Tips for NeRF training data

The NeRF model trains best with between 50-150 images which exhibit minimal scene movement, motion blur or other blurring artefacts. The quality of reconstruction is predicated on COLMAP being able to extract accurate camera parameters from the images.

The `colmap2nerf.py` script assumes that the training images are all pointing approximately at a shared 'point of interest', which it places at the origin. This point is found by taking a weighted average of the closest points of approach between the rays through the central pixel of all pairs of training images. In practice, this means that the script works best when the training images have been captured 'pointing inwards' towards the object of interest, although they do not need to complete a full 360 view of it. Any background visible behind the object of interest will still be reconstructed if `aabb_scale` is set to a number larger than 1, as explained above.

## Python bindings

To conduct controlled experiments in an automated fashion, all features from the interactive testbed (and more!) have Python bindings that can be easily instrumented.
For an example of how the `./build/testbed` application can be implemented and extended from within Python, see `./scripts/run.py`, which supports a superset of the command line arguments that `./build/testbed` does.

Happy hacking!

## Thanks

Many thanks to [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay) and [Andrew Tao](https://developer.nvidia.com/blog/author/atao/) for testing early versions of this codebase and to Arman Toornias and Saurabh Jain for the factory robot dataset.

This project makes use of a number of awesome open source libraries, including:
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for fast CUDA MLP networks
* [tinyexr](https://github.com/syoyo/tinyexr) for EXR format support
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for OBJ format support
* [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
* [Dear ImGui](https://github.com/ocornut/imgui) an excellent immediate mode GUI library
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) a C++ template library for linear algebra
* [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
* and others! See the `dependencies` folder.

Many thanks to the authors of these brilliant projects!

## License

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE.txt) to view a copy of this license.

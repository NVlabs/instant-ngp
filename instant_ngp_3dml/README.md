
### Cloning

Begin by cloning this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/checkandvisit/3dml-instant-ngp.git
$ cd 3dml-instant-ngp
```

# Compilation
Use CMake to generate and build.
```sh
$ cmake . -B build
$ cmake --build build --config RelWithDebInfo -j
```

# Launch the GUI
Test the Fox demo using the command line:
```sh
$ ./instant-ngp data/nerf/fox
```

Visualize any scene and trained weights using the command line:
```sh
$ ./instant-ngp my/nerf/folder --snapshot=my/nerf/snapshot.msgpack
```

# Sync the fork

Update master using latest commits from the upstream NVlabs:master.

```sh
$ git remote add nvidia_ngp https://github.com/NVlabs/instant-ngp.git
$ git fetch nvidia_ngp
$ git checkout master
$ git pull
$ git rebase nvidia_ngp/master
$ git push -f
```

To update the git submodules run:

```sh
$ git submodule update --init --recursive
$ git submodule update --recursive --remote
```

# GUI TIps

- Check if the camera poses are inside the unit cube:
```
Rendering/Debug Visualization/Visualize unit cube
                              Visualize cameras
```

- Move along the input camera path:
```
Rendering/Debug Visualization/Training view
```

- Visualize Input Ground Truth (Color image or Depth)
Press G to toggle Ground Truth Rendering Mode.
Select an alpha smaller than 1.0 to blen the GT into the rendered image.
```
Rendering/NeRF Rendering Options/Ground Truth Render Mode
                                /Ground Truth Alpha
                                /Ground Truth Max Depth
```

### Cloning

Begin by cloning this repository and all its submodules using the following command:
```sh
$ git clone --recursive https://github.com/checkandvisit/3dml-instant-ngp.git
$ cd 3dml-instant-ngp
```

### Compilation
Use CMake to generate and build.
```sh
$ cmake . -B build
$ cmake --build build --config RelWithDebInfo -j
```

### NeRF
Test the Fox demo using the command line:
```sh
$ ./instant-ngp data/nerf/fox
```

Visualize any scene and trained weights using the command line:
```sh
$ ./instant-ngp my/nerf/folder --snapshot=my/nerf/snapshot.msgpack
```


### Sync the fork

Update master using latest commits from the upstream NVlabs:master.

 ```sh
$ git remote add nvidia_ngp https://github.com/NVlabs/instant-ngp.git
$ git fetch nvidia_ngp
$ git checkout master
$ git pull
$ git rebase nvidia_ngp/master
$ git push -f
```

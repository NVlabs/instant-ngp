## VS Code Dev Container

> Basic dev container for running Instant Neural Graphics Primitives.

### Requirements

-   #### **[Docker](https://www.docker.com/get-started)**

-   #### **[VS Code](https://code.visualstudio.com/Download)**

-   #### **[Docker VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)**

### How to build

Without GUI.

```sh
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```

With GUI for Linux distributions.

Allow the docker to connect to the local X server. For example, to allow any remote machine to connect to the local X server, execute the follwing command from the host shell.

```sh
xhost +x
```
Build the project inside the docker container.
```sh
cmake ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```
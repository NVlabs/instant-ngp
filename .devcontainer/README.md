## VS Code Dev Container

> Basic dev container for running Instant Neural Graphics Primitives without GUI.

### Requirements

-   #### **[Docker](https://www.docker.com/get-started)**

-   #### **[VS Code](https://code.visualstudio.com/Download)**

-   #### **[Docker VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)**

### How to build

```sh
cmake -DNGP_BUILD_WITH_GUI=off ./ -B ./build
cmake --build build --config RelWithDebInfo -j 16
```

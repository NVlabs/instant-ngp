# Instant Neural Graphics Primitives ![](https://github.com/NVlabs/instant-ngp/workflows/CI/badge.svg)

<img src="docs/assets_readme/fox.gif" height="342"/> <img src="docs/assets_readme/robot5.gif" height="342"/>

¿Alguna vez quisiste entrenar un modelo NeRF de un zorro en menos de 5 segundos? ¿O volar alrededor de una escena capturada de fotos de un robot de fábrica? ¡Por supuesto que quisiste!

Aquí encontrará una implementación de cuatro __primitivas redes neuronales gráficas__, siendo estas: campos de radiación neural (NeRF), funciones de distancia con signo (SDFs), imagenes neuronales, y volumen neuronal.
En cada caso, entrenamos y renderizamos una MLP con codificación de entrada hash multiresolución usando el framework [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

> __Instant Neural Graphics Primitives with a Multiresolution Hash Encoding__  
> [Thomas Müller](https://tom94.net), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _[arXiv:2201.05989 [cs.CV]](https://arxiv.org/abs/2201.05989), Jan 2022_  
> __[&nbsp;[Project page](https://nvlabs.github.io/instant-ngp)&nbsp;] [&nbsp;[Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)&nbsp;] [&nbsp;[Video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)&nbsp;] [&nbsp;[BibTeX](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.bib)&nbsp;]__

Para preguntas referentes a negocios, por favor visite nuestro sitio web y envíe el formulario: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)


## Requisitos:

- Una tarjeta gráfica NVidia __NVIDIA GPU__; Los núcleos tensoriales aumentan el rendimiento cuando están disponibles. Todos los resultados mostrados provienen de un RTX 3090.
- Un compilador capaz de manejar __C++14__.  Se recomiendan las siguientes opciones y se han probado:
  - __Windows:__ Visual Studio 2019 (necesita también versión mínima WindowsSDK 8.1)
  - __Linux:__ GCC/G++ 7.5 or higher
- __[CUDA](https://developer.nvidia.com/cuda-toolkit) versión v10.2 o mayor__ y __[CMake](https://cmake.org/) v3.19 o mayor__.
- __(opcional) [Python](https://www.python.org/) 3.7 o mayor__ para encuadernaciones interactivas. También, ejecutar `pip install -r requirements.txt`.
  - En algunas máquinas, `pyexr` se rehusa instalar vía `pip`. Se puede resolver instalando OpenEXR desde [aquí](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr).
- __(opcional) [OptiX](https://developer.nvidia.com/optix) 7.3 o mayor__ para un entrenamiento SDF de malla más rápido. Establezca la variable de entorno `OptiX_INSTALL_DIR` en el directorio de instalación si no aparece automáticamente.


Si está utilizando Linux, instale los siguientes paquetes:
```sh
sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
```

También recomendamos instalar [CUDA](https://developer.nvidia.com/cuda-toolkit) y [OptiX](https://developer.nvidia.com/optix) en `/usr/local/` y agregar la instalación CUDA en PATH.
Por ejemplo, si posees CUDA 11.4, agrega lo siguiente a tu `~/.bashrc`
```sh
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```


## Compilación (Windows & Linux)

Comience clonando este repositorio y todos sus submódulos usando el siguiente comando:
```sh
$ git clone --recursive https://github.com/nvlabs/instant-ngp
$ cd instant-ngp
```
Luego, use CMake para compilar el proyecto: (en Windows, esto debe estar en un [símbolo del sistema de desarrollador](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-160#developer_command_prompt))
```sh
instant-ngp$ cmake . -B build
instant-ngp$ cmake --build build --config RelWithDebInfo -j 16
```

Si la compilación falla, consulte [esta lista de posibles soluciones](https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors) antes de abrir un issue.

Si el build tiene éxito, puedes ejecutar el software vía `build/testbed` como ejecutable o usar el comando `scripts/run.py`.

Si falla la detección automática de la arquitectura de la GPU (como puede suceder si tiene varias GPU instaladas), configure la variable de entorno `TCNN_CUDA_ARCHITECTURES` para la GPU que le gustaría usar. La siguiente tabla enumera los valores para las GPU comunes. Si su GPU no está en la lista, consulte [aquí una lista exhaustiva](https://developer.nvidia.com/cuda-gpus).

| RTX 30X0 | A100 | RTX 20X0 | TITAN V / V100 | GTX 10X0 / TITAN Xp | GTX 9X0 | K80 |
|----------|------|----------|----------------|---------------------|---------|-----|
|       86 |   80 |       75 |             70 |                  61 |      52 |  37 |



## Entrenamiento interactivo y renderizado

<img src="docs/assets_readme/testbed.png" width="100%"/>

Este código base viene con un banco de pruebas interactivo que incluye muchas características más allá de nuestra publicación académica:
- Funciones de entrenamiento adicionales, como optimización de elementos extrínsecos e intrínsecos.
- Cubos de marcha para conversión `NeRF->Mesh` y `SDF->Mesh`.
- Un editor de ruta de cámara basado en spline para crear videos.
- Visualizaciones de depuración de las activaciones de cada entrada y salida de neurona.
- Y muchas más configuraciones específicas de tareas.
- Vea también nuestro [video de demostración de un minuto de la herramienta](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4).



### NeRF fox

Se proporciona una escena de prueba en este repositorio, utilizando una pequeña cantidad de fotogramas de un video de teléfono capturado casualmente.:

```sh
instant-ngp$ ./build/testbed --scene data/nerf/fox
```

<img src="docs/assets_readme/fox.png"/>

Alternativamente, descargue cualquier escena compatible con NeRF (e.g. [de la unidad de autores de NeRF](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)).
Ahora puedes ejecutar:

```sh
instant-ngp$ ./build/testbed --scene data/nerf_synthetic/lego/transforms_train.json
```

Para obtener más información sobre cómo preparar conjuntos de datos para su uso con nuestra implementación NeRF, consulte [este documento](docs/nerf_dataset_tips.md).

### SDF armadillo

```sh
instant-ngp$ ./build/testbed --scene data/sdf/armadillo.obj
```

<img src="docs/assets_readme/armadillo.png"/>

### Imagen de Einstein

```sh
instant-ngp$ ./build/testbed --scene data/image/albert.exr
```

<img src="docs/assets_readme/albert.png"/>

Para reproducir los resultados de gigapíxeles, descargue, por ejemplo, [la imagen de Tokio](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837) y conviértalo a `.bin` usando el script `scripts/image2bin.py`. Este formato personalizado mejora la compatibilidad y la velocidad de carga cuando la resolución es alta. Ahora puedes ejecutar:

```sh
instant-ngp$ ./build/testbed --scene data/image/tokyo.bin
```


### Procesador de volumen

Descarga el [volumen nanovdb para la nube de Disney](https://drive.google.com/drive/folders/1SuycSAOSG64k2KLV7oWgyNWyCvZAkafK?usp=sharing), que se deriva [de aquí](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/) ([CC BY-SA 3.0](https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf)).

```sh
instant-ngp$ ./build/testbed --mode volume --scene data/volume/wdas_cloud_quarter.nvdb
```
<img src="docs/assets_readme/cloud.png"/>


## Enlaces de Python

Para realizar experimentos controlados de manera automatizada, todas las funciones del banco de pruebas interactivo (¡y más!) tienen enlaces de Python que se pueden instrumentar fácilmente.
Para ver un ejemplo de cómo se puede implementar y ampliar la aplicación `./build/testbed` desde Python, consulte `./scripts/run.py`, que admite un superconjunto de argumentos de línea de comandos que `./build/testbed` ` hace.

Happy hacking!


## Solución de problemas de errores de compilación

Antes de seguir investigando, asegúrese de que todos los submódulos estén actualizados e intente compilar de nuevo.
```sh
instant-ngp$ git submodule sync --recursive
instant-ngp$ git submodule update --init --recursive
```
Si __instant-ngp__ aún no se compila, actualice CUDA y su compilador a las últimas versiones que pueda instalar en su sistema. Es fundamental que actualice _ambos_, ya que las versiones más recientes de CUDA no siempre son compatibles con los compiladores anteriores y viceversa.
Si su problema persiste, consulte la siguiente tabla de problemas conocidos.

| Problem | Resolution |
|---------|------------|
| __CMake error:__ No CUDA toolset found / CUDA_ARCHITECTURES is empty for target "cmTC_0c70f" | __Windows:__ the Visual Studio CUDA integration was not installed correctly. Follow [these instructions](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963) to fix the problem without re-installing CUDA. ([#18](https://github.com/NVlabs/instant-ngp/issues/18)) |
| | __Linux:__ Environment variables for your CUDA installation are probably incorrectly set. You may work around the issue using ```cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc``` ([#28](https://github.com/NVlabs/instant-ngp/issues/28)) |
| __CMake error:__ No known features for CXX compiler "MSVC" | Reinstall Visual Studio & make sure you run CMake from a developer shell. ([#21](https://github.com/NVlabs/instant-ngp/issues/21)) |
| __Compile error:__ undefined references to "cudaGraphExecUpdate" / identifier "cublasSetWorkspace" is undefined | Update your CUDA installation (which is likely 11.0) to 11.3 or higher. ([#34](https://github.com/NVlabs/instant-ngp/issues/34) [#41](https://github.com/NVlabs/instant-ngp/issues/41) [#42](https://github.com/NVlabs/instant-ngp/issues/42)) |
| __Compile error:__ too few arguments in function call | Update submodules with the above two `git` commands. ([#37](https://github.com/NVlabs/instant-ngp/issues/37) [#52](https://github.com/NVlabs/instant-ngp/issues/52)) |
| __Python error:__ No module named 'pyngp' | It is likely that CMake did not detect your Python installation and therefore did not build `pyngp`. Check CMake logs to verify this. If `pyngp` was built in a different folder than `instant-ngp/build`, Python will be unable to detect it and you have to supply the full path to the import statement. ([#43](https://github.com/NVlabs/instant-ngp/issues/43)) |

If you cannot find your problem in the table, please feel free to [open an issue](https://github.com/NVlabs/instant-ngp/issues/new) and ask for help.

## Thanks

Muchas gracias a [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay) y [Andrew Tao](https://developer.nvidia.com/blog/author/atao/) por probar las primeras versiones de este código base y a Arman Toorians y Saurabh Jain por el conjunto de datos del robot de fábrica.
También agradecemos a [Andrew Webb](https://github.com/grey-area) por notar que uno de los números primos en el hash espacial no era realmente primo; esto se ha solucionado desde entonces.

Este proyecto hace uso de una serie de impresionantes bibliotecas de código abierto, que incluyen:
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for fast CUDA MLP networks
* [tinyexr](https://github.com/syoyo/tinyexr) for EXR format support
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for OBJ format support
* [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
* [Dear ImGui](https://github.com/ocornut/imgui) an excellent immediate mode GUI library
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) a C++ template library for linear algebra
* [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
* and others! See the `dependencies` folder.

¡Muchas gracias a los autores de estos brillantes proyectos!

## Licencia y Citación

```bibtex
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE.txt) to view a copy of this license.

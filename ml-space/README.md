# Запуск модели на платформе ML Space

## Сборка образа
Для того, чтобы собрать образ для развертывания на платформе ML Space необходимо выполнить следующие команды:

```sh
docker build --no-cache -t jupyter-cuda11.0-tf2.4.0-pt1.7.1-gpu-a100_0.0.80-nerf:1.0.1 -f ml-space/Dockerfile .
docker login cr.msk.sbercloud.ru --username <USERNAME>
docker tag jupyter-cuda11.0-tf2.4.0-pt1.7.1-gpu-a100_0.0.80-nerf:1.0.1 cr.msk.sbercloud.ru/df299eee-bace-4fec-91f6-3b1757324136/jupyter-cuda11.0-tf2.4.0-pt1.7.1-gpu-a100_0.0.80-nerf:1.0.1
docker push cr.msk.sbercloud.ru/df299eee-bace-4fec-91f6-3b1757324136/jupyter-cuda11.0-tf2.4.0-pt1.7.1-gpu-a100_0.0.80-nerf:1.0.1
```

## Установка окружения
После запуска инстанса необходимо настроить окружение следующим образом:

```sh
# clone project
git clone --recursive git@github.com:sberbank-cds-ai/instant-ngp.git
cd instant-ngp

# activate environment
bash 
conda activate nerf

# compile project
TCNN_CUDA_ARCHITECTURES=80
cmake . -B build -DNGP_BUILD_WITH_GUI=OFF
cmake --build build --config RelWithDebInfo -j `nproc`

# install python dependencies
CMAKE_VERSION=3.21.0
OPENCV_VERSION=4.5.5.62
pip install -r requirements.txt
pip install cmake==${CMAKE_VERSION} opencv-python==${OPENCV_VERSION}
```

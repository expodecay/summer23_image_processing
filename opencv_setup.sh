#!/bin/bash

OPENCV_VERSION=4.5.5

SOURCE_LOCATION=/opt

PREFIX_LOCATION=/usr/local

echo "Setting up OpenCV $OPENCV_VERSION..."

sudo apt-get update

sudo apt-get install -y wget ca-certificates

wget https://apt.kitware.com/kitware-archive.sh

chmod +x kitware-archive.sh

sudo ./kitware-archive.sh

rm kitware-archive.sh

sudo apt-get install -y cmake
sudo apt-get install build-essential git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
sudo apt-get install python3.8-dev python3-numpy libtbb2 libtbb-dev -y
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev -y

cd $SOURCE_LOCATION
git clone https://github.com/opencv/opencv.git -b $OPENCV_VERSION
git clone https://github.com/opencv/opencv_contrib.git -b $OPENCV_VERSION

cd opencv

mkdir release

cd release

cmake -D BUILD_TIFF=ON \
-D WITH_CUDA=OFF \
-D ENABLE_AVX=OFF \
-D WITH_OPENGL=OFF \
-D WITH_OPENCL=OFF \
-D WITH_IPP=OFF \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D WITH_EIGEN=OFF \
-D WITH_V4L=OFF \
-D WITH_VTK=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D OPENCV_ENABLE_NONFREE=OFF \
-D BUILD_EXAMPLES=ON \
-D PYTHON_EXECUTABLE=/usr/bin/python3 \
-D PYTHON2_EXECUTABLE=/usr/bin/python \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$PREFIX_LOCATION \
-D OPENCV_EXTRA_MODULES_PATH=$SOURCE_LOCATION/opencv_contrib/modules $SOURCE_LOCATION/opencv/

make -j`nproc`

make install

ldconfig

echo "Finished installing OpenCV version:"

pkg-config --modversion opencv4

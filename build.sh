#!/bin/bash

module load cuda/7.5 gcc/4.9.2 ffmpeg/2.6.2 opencv/3.1.0 openmpi/1.10.0

mkdir -p build
rm -rf build/*
cd build
cmake ..
make

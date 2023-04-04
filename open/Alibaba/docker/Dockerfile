FROM ubuntu:22.04

RUN apt update

RUN apt install -y software-properties-common gcc gdb linux-tools-generic
RUN apt install -y libtinfo5 llvm-13 clang-13 libclang-13-dev
RUN apt install -y libopenblas-dev libtool swig scons git vim wget ninja-build gfortran numactl curl unzip
RUN apt install -y libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev automake

RUN apt-get install liblzma-dev -y 

RUN apt install libssl-dev

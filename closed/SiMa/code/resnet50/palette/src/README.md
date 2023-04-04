
# To compile

There is a dependency on the loadgen library, at this point only cross compilation is supported native x86 is not.

To setup cross compiler toolchain and other deps. Download the cross compiler from her below

* https://drive.google.com/file/d/1N6iO5A_sIOCJAH-N2XQHDuLAd6-0ts87/view?usp=share_link

The above would be moved to CI builds soon.

Download the .tar.gz

```shell
$ tar xvf dev-sdk-v.0.2.tar.gz
$ cd sdk
$ sudo ./poky-glibc-x86_64-combo-board-image-cortexa65-combo-board-toolchain-3.4.4.sh
```

This would setup the cross compiler toolchain under /opt/poky/3.4.4/

Then to enable the environment run

```shell
source /opt/poky/3.4.4/environment-setup-cortexa65-poky-linux
```

To compile the loadgen library

```shell
$ cd 3rdparty/loadgen
$ mkdir build
$ cd build
$ cmake ../ && make
```

Once done, compile the plugins

```shell
$ cd src/
$ mkdir build
$ cd build && cmake ../ && make
```

The plugins should be available under

```shell
$ cd build/loadgen_plugin/
```


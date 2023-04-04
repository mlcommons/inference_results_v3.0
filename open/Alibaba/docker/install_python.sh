set -e
set -x

#install python
wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz
tar -zxvf Python-3.9.7.tgz
cd Python-3.9.7
./configure

make
make install
mv /usr/bin/python3 /usr/bin/python3.bak
ln -s /usr/local/bin/python3 /usr/bin/python3
ln -s /usr/local/bin/python3 /usr/bin/python

ln -s /usr/local/bin/pip3 /usr/bin/pip
ln -s /usr/local/bin/pip3 /usr/bin/pip3

rm -f /usr/bin/perf && ln -s /usr/lib/linux-tools/*/perf /usr/bin/perf
rm -rf /usr/bin/lsb_release





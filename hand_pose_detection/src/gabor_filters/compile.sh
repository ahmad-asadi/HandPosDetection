#! /bin/bash
rm -r build
mkdir build
cd build 
cmake ..
make
./main ../../../dataset/NUS/NUS/Train/
echo "Press [ENTER] to run algorithm on test data>>"
read r
./main ../../../dataset/NUS/NUS/Test/
cd ../

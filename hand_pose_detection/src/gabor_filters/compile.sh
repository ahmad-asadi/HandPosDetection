rm -r build
mkdir build
cd build 
cmake ..
make
./main ../../../dataset/NUS/NUS/Train/
cd ../

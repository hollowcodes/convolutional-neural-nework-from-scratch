
all:
	c++ -Isrc -fPIC -I/usr/include/python3.8 -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` cnn_operations.cpp -o cnn_operations`python3-config --extension-suffix`

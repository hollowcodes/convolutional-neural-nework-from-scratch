
#include <iostream>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


py::array_t<double> convolution2d(py::array_t<double> matrix, py::array_t<double> kernel) {
        auto matrix_buf = matrix.mutable_unchecked<2>();
        auto kernel_buf = kernel.mutable_unchecked<2>();

        int rows = matrix_buf.shape(0) - 2;
        int cols = matrix_buf.shape(1) - 2;

        py::array_t<double> result = py::array_t<double>({rows, cols});
        auto result_buf = result.mutable_unchecked<2>();

        for(ssize_t k_i=0; k_i<rows; k_i++) {
                for(ssize_t k_j=0; k_j<cols; k_j++) {

                        float sum = 0;
                        for(ssize_t i=0; i<kernel_buf.shape(0); i++) {
                                for(ssize_t j=0; j<kernel_buf.shape(0); j++) {
                                        sum += matrix_buf(i+k_i, j+k_j) * kernel_buf(i, j);
                                }
                        }
                        result_buf(k_i, k_j) = sum;
                }
        }

        return result;
}

py::array_t<double> maxpool2d(py::array_t<double> matrix, int pooling_size) {
        auto matrix_buf = matrix.mutable_unchecked<2>();

        int rows = matrix_buf.shape(0);
        int cols = matrix_buf.shape(1);

        int to_pool_rows = std::ceil(rows/pooling_size);
        int to_pool_cols = std::ceil(cols/pooling_size);

        py::array_t<double> result = py::array_t<double>({to_pool_rows, to_pool_cols});
        auto result_buf = result.mutable_unchecked<2>();

        for(ssize_t k_i=0; k_i<to_pool_rows; k_i++) {
                for(ssize_t k_j=0; k_j<to_pool_cols; k_j++) {

                        float maximum = 0.0;
                        for(ssize_t i=0; i<pooling_size; i++) {
                                for(ssize_t j=0; j<pooling_size; j++) {
                                        int row_index = i + (k_i * pooling_size);
                                        int col_index = j + (k_j * pooling_size);

                                        double element = matrix_buf(row_index, col_index);

                                        if(element > maximum) {
                                                maximum = element;
                                        }
                                }
                        }
                        result_buf(k_i, k_j) = maximum;
                }
        }

        return result;
}


PYBIND11_MODULE(cnn_operations, m) {
        m.doc() = "C++ accelerated operations for convolutional-neural-networks.";

        m.def("convolution2d", &convolution2d, "Convolutes a 2d matrix with a 2d kernel.");
        m.def("maxpool2d", &maxpool2d, "Applies maxpooling to a 2d matrix.");

}


/*
py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
        auto buf1 = input1.request();
        auto buf2 = input2.request();

        if (buf1.size != buf2.size) {
                throw std::runtime_error("Input shapes must match");
        }
        
        int result_shape[2] = {buf1.shape[0], buf2.shape[1]};

        py::array_t<double> result = py::array_t<double>(result_shape);

        auto buf3 = result.request();

        double *ptr1 = (double *) buf1.ptr;
        double *ptr2 = (double *) buf2.ptr;
        double *ptr3 = (double *) buf3.ptr;

        int columns = buf1.shape[0];
        int rows = buf1.shape[1];

        for (size_t i=0; i<columns; i++) {
                for (size_t j=0; j<rows; j++) {
                        ptr3[i*rows + j] = 4;//ptr1[i*rows+ j] + ptr2[i*rows+ j];
                }
        }

        // reshape array to match input shape
        //result.resize({X,Y});

        return result;
}
*/
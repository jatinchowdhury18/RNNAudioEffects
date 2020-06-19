include_directories(../../modules/Eigen)

set(CMAKE_CXX_STANDARD 17)

add_library(MLUtils STATIC gru.h gru.cpp gru_eigen.h dense.h dense_eigen.h activation.h Model.h Layer.h)

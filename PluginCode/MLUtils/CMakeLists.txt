include_directories(../../modules/Eigen)

add_library(MLUtils STATIC gru.h gru.cpp gru_eigen.h dense.h dense_eigen.h activation.h Model.h Layer.h)

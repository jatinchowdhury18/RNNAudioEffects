#ifndef DENSE_H_INCLUDED
#define DENSE_H_INCLUDED

#include <algorithm>
#include <numeric>
#include "Layer.h"

#ifdef USE_EIGEN_DENSE
#include <Eigen/Eigen>
#endif

template<typename T>
class Dense1
{
public:
    Dense1 (size_t in_size) :
        in_size (in_size)
    {
#ifdef USE_EIGEN_DENSE
        weightsVec.resize (in_size);
        inVec.resize (in_size);
#else
        weights = new T[in_size];
#endif
    }

    ~Dense1()
    {
#ifndef USE_EIGEN_DENSE
        delete[] weights;
#endif
    }

    inline T forward(const T* input)
    {
#ifdef USE_EIGEN_DENSE
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> (input, in_size, 1);
        return inVec.dot(weightsVec) + bias;
#else
        return std::inner_product(weights, weights + in_size, input, (T) 0) + bias;
#endif
    }

    void setWeights(const T* newWeights)
    {
        for(int i = 0; i < in_size; ++i)
        {
#ifdef USE_EIGEN_DENSE
            weightsVec (i, 0) = newWeights[i];
#else
            weights[i] = newWeights[i];
#endif
        }
    }

    void setBias(T b) { bias = b; }

private:
    const size_t in_size;
    T bias;

#ifdef USE_EIGEN_DENSE
    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> weightsVec;
#else
    T* weights;
#endif
};

template<typename T>
class Dense : public Layer<T>
{
public:
    Dense (size_t in_size, size_t out_size) :
        Layer<T> (in_size, out_size)
    {
        subLayers = new Dense1<T>*[out_size];
        for (int i = 0; i < out_size; ++i)
            subLayers[i] = new Dense1<T> (in_size);
    }

    virtual ~Dense()
    {
        for (int i = 0; i < out_size; ++i)
            delete subLayers[i];

        delete[] subLayers;
    }

    inline void forward (const T* input, T* out) override
    {
        for (int i = 0; i < out_size; ++i)
        {
            out[i] = subLayers[i]->forward (input);
        }
    }

    void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            subLayers[i]->setWeights (newWeights[i]);
    }

    void setBias(T* b)
    {
        for(int i = 0; i < out_size; ++i)
            subLayers[i]->setBias (b[i]);
    }

private:
    Dense1<T>** subLayers;
};

#endif // DENSE_H_INCLUDED

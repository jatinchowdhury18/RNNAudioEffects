#ifndef GRU_H_INCLUDED
#define GRU_H_INCLUDED

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

#ifdef USE_EIGEN
#include "gru_eigen.h"
#else
#include "Layer.h"

template<typename T>
class GRULayer : public Layer<T>
{
public:
    GRULayer (size_t in_size, size_t out_size) :
        Layer<T> (in_size, out_size),
        zWeights (in_size, out_size),
        rWeights (in_size, out_size),
        cWeights (in_size, out_size)
    {
        ht1 = new T[out_size];
        zVec = new T[out_size];
        rVec = new T[out_size];
        cVec = new T[out_size];
    }

    virtual ~GRULayer()
    {
        delete[] ht1;
        delete[] zVec;
        delete[] rVec;
        delete[] cVec;
    }

    void reset()
    {
        std::fill(ht1, ht1 + out_size, (T) 0);
    }

    inline void forward(const T* input, T* h) override
    {
        for(int i = 0; i < out_size; ++i)
        {
            zVec[i] = sigmoid(vMult(zWeights.W[i], input, in_size) + vMult(zWeights.U[i], ht1, out_size) + zWeights.b[0][i] + zWeights.b[1][i]);
            rVec[i] = sigmoid(vMult(rWeights.W[i], input, in_size) + vMult(rWeights.U[i], ht1, out_size) + rWeights.b[0][i] + rWeights.b[1][i]);
            cVec[i] = std::tanh(vMult(cWeights.W[i], input, in_size) + rVec[i] * (vMult(cWeights.U[i], ht1, out_size) + cWeights.b[1][i]) + cWeights.b[0][i]);
            h[i] = ((T) 1 - zVec[i]) * cVec[i] + zVec[i] * ht1[i];
        }
    
        std::copy(h, h + out_size, ht1);
    }

    void setWVals(T** wVals)
    {
        for(int i = 0; i < in_size; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                zWeights.W[k][i] = wVals[i][k];
                rWeights.W[k][i] = wVals[i][k+out_size];
                cWeights.W[k][i] = wVals[i][k+out_size*2];
            }
        }
    }

    void setUVals(T** uVals)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                zWeights.U[k][i] = uVals[i][k];
                rWeights.U[k][i] = uVals[i][k+out_size];
                cWeights.U[k][i] = uVals[i][k+out_size*2];
            }
        }
    }

    void setBVals(T** bVals)
    {
        for(int i = 0; i < 2; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                zWeights.b[i][k] = bVals[i][k];
                rWeights.b[i][k] = bVals[i][k+out_size];
                cWeights.b[i][k] = bVals[i][k+out_size*2];
            }
        }
    }

    inline T vMult(const T* arg1, const T* arg2, size_t dim)
    {
        return std::inner_product(arg1, arg1 + dim, arg2, (T) 0);
    }

    inline T sigmoid(T value)
    {
        return (T) 1 / ((T) 1 + std::exp(-value));
    }

private:
    T* ht1;

    struct WeightSet
    {
        WeightSet (size_t in_size, size_t out_size) :
            out_size (out_size)
        {
            W = new T*[out_size];
            U = new T*[out_size];
            b[0] = new T[out_size];
            b[1] = new T[out_size];

            for (size_t i = 0; i < out_size; ++i)
            {
                W[i] = new T[in_size];
                U[i] = new T[out_size];
            }
        }

        ~WeightSet()
        {
            delete[] b[0];
            delete[] b[1];

            for (size_t i = 0; i < out_size; ++i)
            {
                delete[] W[i];
                delete[] U[i];
            }

            delete[] W;
            delete[] U;
        }

        T** W;
        T** U;
        T* b[2];
        const size_t out_size;
    };

    WeightSet zWeights;
    WeightSet rWeights;
    WeightSet cWeights;

    T* zVec;
    T* rVec;
    T* cVec;
};
#endif

#endif // GRU_H_INCLUDED

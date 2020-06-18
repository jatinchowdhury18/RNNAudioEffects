#ifndef GRUEIGEN_H_INCLUDED
#define GRUEIGEN_H_INCLUDED

#include <Eigen/Eigen>
#include "Layer.h"

template<typename T>
class GRULayer : public Layer<T>
{
public:
    GRULayer (size_t in_size, size_t out_size) :
        Layer<T> (in_size, out_size)
    {
        wVec_z.resize (out_size, in_size);
        wVec_r.resize (out_size, in_size);
        wVec_c.resize (out_size, in_size);
        uVec_z.resize (out_size, out_size);
        uVec_r.resize (out_size, out_size);
        uVec_c.resize (out_size, out_size);
        bVec_z.resize (out_size, 2);
        bVec_r.resize (out_size, 2);
        bVec_c.resize (out_size, 2);

        ht1.resize (out_size, 1);
        zVec.resize (out_size, 1);
        rVec.resize (out_size, 1);
        cVec.resize (out_size, 1);

        inVec.resize (in_size, 1);
        ones = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones (out_size, 1);
    }

    virtual ~GRULayer()
    {
    }

    void reset()
    {
        std::fill(ht1.data(), ht1.data() + out_size, (T) 0);
    }

    inline void forward(const T* input, T* h) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> (input, in_size, 1);

        zVec = wVec_z * inVec + uVec_z * ht1 + bVec_z.col (0) + bVec_z.col (1);
        rVec = wVec_r * inVec + uVec_r * ht1 + bVec_r.col (0) + bVec_r.col (1);
        sigmoid (zVec);
        sigmoid (rVec);
        
        cVec = wVec_c * inVec + rVec.cwiseProduct (uVec_c * ht1 + bVec_c.col (1)) + bVec_c.col (0);
        cVec = cVec.array().tanh();
        
        ht1 = (ones - zVec).cwiseProduct (cVec) + zVec.cwiseProduct (ht1);
        std::copy (ht1.data(), ht1.data() + out_size, h);
    }

    inline void sigmoid (Eigen::Matrix<T, Eigen::Dynamic, 1>& vector)
    {
        vector = (T) 1 / (((T) -1 * vector.array()).array().exp() + (T) 1);
    }

    void setWVals(T** wVals)
    {
        for(int i = 0; i < in_size; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                wVec_z (k, i) = wVals[i][k];
                wVec_r (k, i) = wVals[i][k+out_size];
                wVec_c (k, i) = wVals[i][k+out_size*2];
            }
        }
    }

    void setUVals(T** uVals)
    {
        for(int i = 0; i < out_size; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                uVec_z (k, i) = uVals[i][k];
                uVec_r (k, i) = uVals[i][k+out_size];
                uVec_c (k, i) = uVals[i][k+out_size*2];
            }
        }
    }

    void setBVals(T** bVals)
    {
        for(int i = 0; i < 2; ++i)
        {
            for(int k = 0; k < out_size; ++k)
            {
                bVec_z (k, i) = bVals[i][k];
                bVec_r (k, i) = bVals[i][k+out_size];
                bVec_c (k, i) = bVals[i][k+out_size*2];
            }
        }
    }

private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> wVec_c;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> uVec_c;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_z;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_r;
    Eigen::Matrix<T, Eigen::Dynamic, 2> bVec_c;

    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> zVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> rVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> cVec;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ones;
};

#endif // GRUEIGEN_H_INCLUDED

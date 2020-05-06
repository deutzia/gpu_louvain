#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

void detect_cuda_init();

#define CUDA_CHECK(error)                                                    \
    {                                                                        \
        auto status = static_cast<cudaError_t>(error);                       \
        if (status != cudaSuccess) {                                         \
            std::cerr << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                      << ", CUDA error:" << cudaGetErrorString(status);      \
            exit(1);                                                         \
        }                                                                    \
    }

template <class C>
void device_set_var(C* var, C val)
{
    CUDA_CHECK(cudaMemcpy(var, &val, sizeof(C), cudaMemcpyHostToDevice));
}

template <class C>
C device_fetch_var(const C* var)
{
    C res;
    CUDA_CHECK(cudaMemcpy(&res, var, sizeof(C), cudaMemcpyDeviceToHost));
    return res;
}

#endif  // __UTILS_H__


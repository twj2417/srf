﻿#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define abs(x) ((x)<0 ? (-x) : (x))

const int GRIDDIM = 32;
const int BLOCKDIM = 1024;
const float PI = 3.1415926;
// First experimental results of time-of-flight
// reconstruction on an LSO PET scanner

__device__ void backproject_device(const float x1_, const float y1_,
                                   const float x2_, const float y2_,
                                   const int nx, const int ny,
                                   const float cx, const float cy,
                                   const float sx, const float sy,
                                   const float tof_value, const float tof_bin, const float
                                   tof_sigma2,
                                   const float vproj, float *image)
{
    const float dx_ = sx / nx;
    const float dx = 1.0f;
    const float dy = sy / ny / dx_;
    const float x1 = (x1_ - cx) / dx_;
    const float x2 = (x2_ - cx) / dx_;
    const float y1 = (y1_ - cy) / dx_;
    const float y2 = (y2_ - cy) / dx_;

    const float xd = x2 - x1;
    const float yd = y2 - y1;

    if (sqrt(xd * xd + yd * yd) < 10.0f) {return;}

    const float nx2 = nx / 2.0f;
    const float ny2 = ny / 2.0f;

    const float L = sqrt(xd * xd + yd * yd);
    const float ratio1 = (1.0 - tof_value / dx_ / L) / 2;
    const float tof_bin_ = tof_bin / dx_;
    const float tof_sigma2_ = tof_sigma2 / dx_ / dx_;
    if (abs(xd) > abs(yd))
    {
        float ky = yd / xd;

        for (int ix = 0; ix < nx; ++ix)
        {
            float xc = (ix - nx / 2 + 0.5) * dx;
            float d2_tof = ((xc - x1) / (x2 - x1) - ratio1) * L;
            float tof_sigma2_expand = tof_sigma2_ + (tof_bin_ * tof_bin_) / 12;
            float t2 = d2_tof * d2_tof / tof_sigma2_expand;
            float w_tof = tof_bin_ * exp(-0.5 * t2) / sqrt(2.0 * PI * tof_sigma2_expand);
            float xx1 = ix - nx2;
            float xx2 = xx1 + 1.0f;
            float yy1, yy2;

            if (ky >= 0.0f)
            {
                yy1 = (y1 + ky * (xx1 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx2 - x1)) / dy + ny2;

            }
            else
            {
                yy1 = (y1 + ky * (xx2 - x1)) / dy + ny2;
                yy2 = (y1 + ky * (xx1 - x1)) / dy + ny2;
            }
            int cy1 = (int)floor(yy1);
            int cy2 = (int)floor(yy2);

            if (cy1 == cy2)
            {
                if (0 <= cy1 && cy1 < ny)
                {
                    float weight = sqrt(1 + ky * ky) * dx_;
                    atomicAdd(image + ix + cy1 * nx, vproj * weight * w_tof);
                }
            }
            else
            {
                float ry = (cy2 - yy1) / (yy2 - yy1);
                if (cy1 >= 0)
                {
                    float weight = ry * sqrt(1 + ky * ky) * dx_;
                    atomicAdd(image + ix + cy1 * nx, vproj * weight * w_tof);
                }


                if (cy2 < ny)
                {
                    float weight = (1 - ry) * sqrt(1 + ky * ky) * dx_;
                    atomicAdd(image + ix + cy2 * nx, vproj * weight * w_tof);
                }
            }
        }
    }
    else
    {
        float kx = xd / yd;

        for (int iy = 0; iy < ny; ++iy)
        {
            float yc = (iy - ny / 2 + 0.5) * dy;
            float d2_tof = ((yc - y1) / (y2 - y1) - ratio1) * L;
            float tof_sigma2_expand = tof_sigma2_ + (tof_bin_ * tof_bin_) / 12;
            float t2 = d2_tof * d2_tof / tof_sigma2_expand;
            float w_tof = tof_bin_ * exp(-0.5 * t2) / sqrt(2.0 * PI * tof_sigma2_expand);
            float yy1 = iy - ny2;
            float yy2 = yy1 + 1.0f;
            float xx1, xx2;

            if (kx >= 0.0f)
            {
                xx1 = (x1 + kx * (yy1 - y1)) + nx2;
                xx2 = (x1 + kx * (yy2 - y1)) + nx2;
            }
            else
            {
                xx1 = (x1 + kx * (yy2 - y1)) + nx2;
                xx2 = (x1 + kx * (yy1 - y1)) + nx2;
            }
            int cx1 = (int)floor(xx1);
            int cx2 = (int)floor(xx2);

            if (cx1 == cx2)
            {
                float weight = sqrt(1 + kx * kx) * dx_;
                atomicAdd(image + cx1 + iy * nx, vproj * weight * w_tof);
            }
            else
            {
                float rx = (cx2 - xx1) / (xx2 - xx1);
                if (cx1 >= 0)
                {
                    float weight = rx * sqrt(1 + kx * kx) * dx_;
                    atomicAdd(image + cx1 + iy * nx, vproj * weight * w_tof);
                }

                if (cx2 < nx)
                {
                    float weight = (1 - rx) * sqrt(1 + kx * kx) * dx_;
                    atomicAdd(image + cx2 + iy * nx, vproj * weight * w_tof);
                }
            }
        }
    }
}

__global__ void
BackProjectKernel(const float *x1, const float *y1,
                    const float *x2, const float *y2,
                    const int gx, const int gy,
                    const float cx, const float cy,
                    const float sx, const float sy,
                    const int num_events,
                    const float * tof_values, const float tof_bin, const float tof_sigma2,
                    const float *projection_value, float *image_data)
{
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid >= num_events) {return;}

        backproject_device(x1[tid], y1[tid],
                           x2[tid], y2[tid],
                           gx, gy,
                           cx, cy,
                           sx, sy,
                           tof_values[tid], tof_bin, tof_sigma2,
                           projection_value[tid], image_data);

    }
}


void backprojection(const float *x1, const float *y1,
                    const float *x2, const float *y2,
                    const float *vproj,
                    const int *grid, const float *center, const float *size,
                    const float * tof_values, const float tof_bin, const float tof_sigma2,
                    float *image, const int num_events)
{
    int grid_cpu[2];
    float center_cpu[2];
    float size_cpu[2];
    cudaMemcpy(grid_cpu, grid, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1]; //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1];
    BackProjectKernel<<<GRIDDIM, BLOCKDIM>>>(x1, y1,
                                             x2, y2,
                                             gx, gy,
                                             cx, cy,
                                             sx, sy,
                                             num_events,
                                             tof_values, tof_bin, tof_sigma2,
                                             vproj, image);
}

#endif

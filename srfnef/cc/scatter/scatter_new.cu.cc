#define GOOGLE_CUDA 1
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "cuda.h"
#include "cuda_runtime.h"
#define ABS(x) ((x) < 0 ? (-x) : (x))

const int GRIDDIM = 32;
const int BLOCKDIM = 1024;
const float PI = 3.1415926;

__device__ float length_(const float x1, const float y1, const float z1,
                         const float x2, const float y2, const float z2)
{
    return (float)sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
}

__device__ float efficiency(const float low_eng, const float high_eng, const float res_eng,
                            const float energy)
{
    float eff = 0;
    for (float en = low_eng; en < high_eng; en += 5)
    {
        eff += expf(-(en - energy) * (en - energy) / 2 / res_eng / res_eng);
    }
    return eff / sqrt(2 * PI) / res_eng;
}

__device__ float project_area(const float angle_per_block, const float crystal_area,
                              const float x1, const float y1, const float z1,
                              const float x2, const float y2, const float z2)
{
    float theta_normal = round(atan2(y2, x2) / angle_per_block) * angle_per_block;
    return ABS((x1 - x2) * cos(theta_normal) + (y1 - y2) * sin(theta_normal)) / length_(x1, y1, z1, x2, y2, z2) * crystal_area;
}

__device__ float fkn(const int ix, const int iy, const int iz,
                     const int nx, const int ny, const int nz,
                     const float cos_asb,
                     const float *umap)
{
        float sigma = PI * (40.0 / 9.0 - 3.0 * logf(3.0));
        float differential_value = 1 / 2 / (2 - cos_asb) / (2 - cos_asb) * ((1.0 + cos_asb * cos_asb) + (1.0 - cos_asb) * (1.0 - cos_asb) / (2 - cos_asb));
        float total_cross = 1000 * (umap[ix + nx * iy + nx * ny * iz] - 0.0092) / 0.0092 * 0.00000957 + 0.0095784;
        return differential_value / sigma * total_cross;
    }
}

__device__ void single_lor_scatter(const float x1, const float y1, const float z1,
                                   const float x2, const float y2, const float z2,
                                   const int ind1, const int ind2,
                                   const float *umap_project, const float *image_project,
                                   const float *umap,
                                   const int nx, const int ny, const int nz,
                                   const float cx, const float cy, const float cz,
                                   const float sx, const float sy, const float sz,
                                   const int smp_x, const int smp_y, const int smp_z,
                                   const float low_eng, const float high_eng, const float res_eng,
                                   const float angle_per_block, const float crystal_area,
                                   float *vproj)
{
    float scatter_lor = 0.0f;
    float usx = sx / float(nx); // image unit size in x
    float usy = sy / float(ny);
    float usz = sz / float(nz);
    int nx_smp = nx / smp_x;
    int ny_smp = ny / smp_y;
    int nz_smp = nz / smp_z;
    int n_smp = nx_smp * ny_smp * nz_smp;

    for (int ix = 0; ix < nx; ix += smp_x)
    {
        for (int iy = 0; iy < ny; iy += smp_y)
        {
            for (int iz = 0; iz < nz; iz += smp_z)
            {
                int id1 = ix + iy * nx_smp + iz * nx_smp * ny_smp + ind1 * n_smp;
                int id2 = ix + iy * nx_smp + iz * nx_smp * ny_smp + ind2 * n_smp;
                float xp = (ix + 0.5) * 4 + cx - 100; // scatter point coordinate
                float yp = (iy + 0.5) * 4 + cy - 100;
                float zp = (iz + 0.5) * 4 + cz - 104;
                float dist_pnt_fst = length_(x1, y1, z1, xp, yp, zp);
                float dist_pnt_snd = length_(x2, y2, z2, xp, yp, zp);
                float cos_theta = ((x2 - xp) * (x1 - xp) + (y2 - yp) * (y1 - yp) + (z2 - zp) * (z1 - zp)) /
                                  dist_pnt_fst / dist_pnt_snd;
                float fkn_value = fkn(ix, iy, iz, nx, ny, nz, cos_theta, umap);

                scatter_lor += fkn_value;
                float scatter_energy = 511 / (2 - cos_theta);

                float atten_pnt_fst = umap_project[id1];
                float atten_pnt_snd = umap_project[id2];
                float emiss_pnt_fst = image_project[id1];
                float emiss_pnt_snd = image_project[id2];

                float I_A = efficiency(low_eng, high_eng, res_eng, 511.0) * efficiency(low_eng, high_eng, res_eng, scatter_energy) * expf(-atten_pnt_fst - atten_pnt_snd) * emiss_pnt_fst;
                float I_B = efficiency(low_eng, high_eng, res_eng, 511.0) * efficiency(low_eng, high_eng, res_eng, scatter_energy) * expf(-atten_pnt_fst - atten_pnt_snd) * emiss_pnt_snd;

                float sigma_as = project_area(angle_per_block, crystal_area, xp, yp, zp, x1, y1, z1);
                float sigma_bs = project_area(angle_per_block, crystal_area, xp, yp, zp, x2, y2, z2);
                scatter_lor += (I_A + I_B) * fkn_value * sigma_as * sigma_bs / 4 / PI / dist_pnt_fst / dist_pnt_fst / dist_pnt_snd / dist_pnt_snd;
            }
        }
    }
    vproj[0] = scatter_lor * smp_x * smp_y * smp_z * 4 * 4 * 4;
}


__device__ void single_lor_scale(const float x1, const float y1, const float z1,
                                 const float x2, const float y2, const float z2,
                                 const float angle_per_block, const float crystal_area,
                                 const float epsilon_ab,
                                 float *lor_scales)
{
    const float area_ab = project_area(angle_per_block, crystal_area, x1, y1, z1, x2, y2, z2);
    const float area_ba = project_area(angle_per_block, crystal_area, x2, y2, z2, x1, y1, z1);
    const float length_ab = length_(x1, y1, z1, x2, y2, z2);
    lor_scales[0] = 4.0f * PI * length_ab * length_ab / epsilon_ab / epsilon_ab; // / area_ab / area_ba;
}

__global__ void
ScatterKernel(const float *x1, const float *y1, const float *z1,
              const float *x2, const float *y2, const float *z2,
              const int *ind1, const int *ind2,
              const float *umap_project, const float *image_project,
              const float *umap,
              const int nx, const int ny, const int nz,
              const float cx, const float cy, const float cz,
              const float sx, const float sy, const float sz,
              const int smp_x, const int smp_y, const int smp_z,
              const float low_eng, const float high_eng, const float res_eng,
              const float angle_per_block, const float crystal_area, const int num_events,
              float *vproj)
{
    int step = blockDim.x * gridDim.x;
    // int jid = threadIdx.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid >= num_events)
        {
            return;
        }
        single_lor_scatter(x1[tid], y1[tid], z1[tid],
                           x2[tid], y2[tid], z2[tid],
                           ind1[tid], ind2[tid], umap_project, image_project, umap,
                           nx, ny, nz, cx, cy, cz, sx, sy, sz, smp_x, smp_y, smp_z,
                           low_eng, high_eng, res_eng, angle_per_block, crystal_area,
                           vproj + tid);
    }
}

__global__ void
ScaleKernel(const float *x1, const float *y1, const float *z1,
            const float *x2, const float *y2, const float *z2,
            const float angle_per_block, const float crystal_area, const float epsilon_ab,
            const int num_events,
            float *lor_scales)
{
    int step = blockDim.x * gridDim.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < (num_events + step); tid += step)
    {
        if (tid >= num_events)
        {
            return;
        }
        single_lor_scale(x1[tid], y1[tid], z1[tid],
                         x2[tid], y2[tid], z2[tid],
                         angle_per_block, crystal_area, epsilon_ab,
                         lor_scales + tid);
    }
}

void scatter_loop_all_lors(const float *x1, const float *y1, const float *z1,
                           const float *x2, const float *y2, const float *z2,
                           const int *ind1, const int *ind2,
                           const float *umap_project, const float *image_project,
                           const float *umap,
                           const int *grid, const float *center, const float *size, const int *smp,
                           const float low_eng, const float high_eng, const float res_eng,
                           const float angle_per_block, const float crystal_area,
                           const int num_events,
                           float *vproj)
{
    int grid_cpu[3];
    int smp_cpu[3];
    float center_cpu[3];
    float size_cpu[3];
    cudaMemcpy(grid_cpu, grid, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(center_cpu, center, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size_cpu, size, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(smp_cpu, smp, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    int gx = grid_cpu[0], gy = grid_cpu[1], gz = grid_cpu[2];         //number of meshes
    float cx = center_cpu[0], cy = center_cpu[1], cz = center_cpu[2]; // position of center
    float sx = size_cpu[0], sy = size_cpu[1], sz = size_cpu[2];
    int smpx = smp_cpu[0], smpy = smp_cpu[1], smpz = smp_cpu[2]; // undersampling rate
    ScatterKernel<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                         x2, y2, z2,
                                         ind1, ind2,
                                         umap_project, image_project, umap,
                                         gx, gy, gz,
                                         cx, cy, cz,
                                         sx, sy, sz,
                                         smpx, smpy, smpz,
                                         low_eng, high_eng, res_eng,
                                         angle_per_block, crystal_area,
                                         num_events,
                                         vproj);
    cudaDeviceSynchronize();
}

void scale_loop_all_lors(const float *x1, const float *y1, const float *z1,
                         const float *x2, const float *y2, const float *z2,
                         const float angle_per_block, const float crystal_area,
                         const float epsilon_ab,
                         const int num_events,
                         float *lor_scales)
{
    ScaleKernel<<<GRIDDIM, BLOCKDIM>>>(x1, y1, z1,
                                       x2, y2, z2,
                                       angle_per_block, crystal_area, epsilon_ab, num_events,
                                       lor_scales);
    cudaDeviceSynchronize();
}

#endif

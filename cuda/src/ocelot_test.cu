#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cuda_help.h"

#define BLOCK_DIM 32

__global__ void kernel(int *vec_a, int *vec_b, int vec_dim)
{

	__shared__ int sh_vec_a[BLOCK_DIM + 2];

	int grid_dim = gridDim.x * blockDim.x;

	int step = 0;
	int max_steps = (vec_dim + (2 * gridDim.x) + grid_dim) / grid_dim;
	while (step < max_steps)
	{
		int gl_ind = (step * grid_dim) + blockDim.x * blockIdx.x + threadIdx.x;
		int stride_ind = gl_ind - blockIdx.x * 2 - step * gridDim.x * 2;
		// int stride_ind = gl_ind;

		if (stride_ind < vec_dim)
		{
			sh_vec_a[threadIdx.x] = vec_a[stride_ind];
		}
		__syncthreads();

		if (stride_ind < vec_dim)
		{
			if (threadIdx.x < BLOCK_DIM - 2)
			{
				int value = sh_vec_a[threadIdx.x] + sh_vec_a[threadIdx.x + 1] + sh_vec_a[threadIdx.x + 2];

				vec_b[stride_ind] = value;
			}
		}

		__syncthreads();

		step++;
	}

	return;
}

int main(void)
{

	int vec_dim = 1000;

	int *vec_a = (int *)malloc(vec_dim * sizeof(int));
	int *vec_b = (int *)malloc(vec_dim * sizeof(int));

	init_vec(vec_a, vec_dim, 1);
	init_vec(vec_b, vec_dim, 0);

	int *dev_vec_a;
	int *dev_vec_b;
	cuda_err(cudaMalloc((void **)&dev_vec_a, vec_dim * sizeof(vec_dim)),
			 "vec_a malloc");
	cuda_err(cudaMalloc((void **)&dev_vec_b, vec_dim * sizeof(vec_dim)),
			 "vec_b malloc");

	cuda_err(cudaMemcpy(dev_vec_a, vec_a, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec_a memcpy to device");

	cuda_err(cudaMemcpy(dev_vec_b, vec_b, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec_b memcpy to device");

	int grid_dim = (vec_dim + BLOCK_DIM) / BLOCK_DIM;

	kernel<<<1, BLOCK_DIM>>>(dev_vec_a, dev_vec_b, vec_dim);

	printf("Infinite kernel is running in the background ...  ");

	cuda_err(cudaMemcpy(vec_a, dev_vec_a, vec_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "vec_a memcpy to device");

	cuda_err(cudaMemcpy(vec_b, dev_vec_b, vec_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "vec_b memcpy to device");

	cudaFree(dev_vec_a);
	cudaFree(dev_vec_b);

	printf("Just the samples (10) ... \n");
	print_vec(vec_a, 10);
	print_vec(vec_b, vec_dim);

	free(vec_a);
	free(vec_b);

	return 0;
}

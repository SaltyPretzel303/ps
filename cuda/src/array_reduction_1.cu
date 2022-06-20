#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cuda_help.h"

#define BLOCK_DIM 32

__global__ void sum_vec(int *vec, int *part_sum, int dim)
{
	__shared__ int sh_vec[BLOCK_DIM];

	int gl_ind = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_stride = blockDim.x * gridDim.x;

	if (gl_ind < dim)
	{
		sh_vec[threadIdx.x] = vec[gl_ind] + vec[gl_ind + grid_stride];
	}

	__syncthreads();

	int stride = blockDim.x / 2;

	while (stride > 0)
	{
		if (threadIdx.x <= stride)
		{
			int a = sh_vec[threadIdx.x];
			int b = sh_vec[threadIdx.x + stride];

			sh_vec[threadIdx.x] = a + b;
		}

		stride /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		part_sum[blockIdx.x] = sh_vec[0];
	}
}

int main(void)
{

	int vec_dim = 1000;

	int *vec = (int *)malloc(vec_dim * sizeof(int));
	init_vec(vec, vec_dim, 1);

	int check_sum = vec_dim;

	int *dev_vec;
	cuda_err(cudaMalloc((void **)&dev_vec, vec_dim * sizeof(int)),
			 "vec cudaMalloc");

	int grid_dim = (vec_dim + BLOCK_DIM) / BLOCK_DIM / 2;

	int *dev_part_vec;
	cuda_err(cudaMalloc((void **)&dev_part_vec, grid_dim * sizeof(int)),
			 "part_res_vec cudaMalloc");

	cuda_err(cudaMemcpy(dev_vec, vec, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec cudaMemcpy to device");

	int sum;
	int *dev_sum;

	cuda_err(cudaMalloc((void **)&dev_sum, 1 * sizeof(int)),
			 "sum cudaMalloc");

	sum_vec<<<grid_dim, BLOCK_DIM>>>(dev_vec, dev_part_vec, vec_dim);

	// dubug print

	// int *test_vec = (int *)malloc(grid_dim * sizeof(int));
	// cuda_err(cudaMemcpy(test_vec, dev_part_vec, grid_dim * sizeof(int), cudaMemcpyDeviceToHost),
	// 		 "debug_vec cudaMemcpy device to host");
	// print_vec(test_vec, grid_dim);

	// return 1;

	// dubug print

	sum_vec<<<1, grid_dim>>>(dev_part_vec, dev_sum, grid_dim);

	cuda_err(cudaMemcpy(&sum, dev_sum, 1 * sizeof(int), cudaMemcpyDeviceToHost),
			 "sum cudaMemcpy to host");

	cudaFree(dev_vec);
	cudaFree(dev_part_vec);
	cudaFree(dev_sum);

	if (sum == check_sum)
	{
		printf("Success: %d ... \n", sum);
	}
	else
	{
		printf("Failed -> check_sum: %d  sum: %d  diff: %d ... \n",
			   check_sum, sum, check_sum - sum);
	}

	free(vec);

	cudaDeviceReset();

	return 0;
}
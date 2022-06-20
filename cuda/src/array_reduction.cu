#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cuda_help.h"

#define BLOCK_DIM 32

__global__ void sum_vec(int *vec, int *part_sum, int dim)
{
	__shared__ int sh_vec[BLOCK_DIM];

	int gl_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (gl_id < dim)
	{
		sh_vec[threadIdx.x] = vec[gl_id];
	}
	__syncthreads();

	int stride = 2;

	while (stride <= BLOCK_DIM)
	{
		int ind = threadIdx.x * stride;

		if (ind < BLOCK_DIM)
		{
			int a = sh_vec[ind];
			int b = sh_vec[ind + stride / 2];
			sh_vec[ind] = a + b;
		}

		stride *= 2;

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

	int check_sum = vec_dim; // sum of the elements sould be equal to vec_dim

	int *dev_vec;
	cuda_err(cudaMalloc((void **)&dev_vec, vec_dim * sizeof(int)),
			 "vec cudaMalloc");

	int dev_part_dim = (vec_dim + BLOCK_DIM) / BLOCK_DIM; // basically number of blocks
	int *dev_part_vec;
	cuda_err(cudaMalloc((void **)&dev_part_vec, dev_part_dim * sizeof(int)),
			 "part_res_vec cudaMalloc");

	cuda_err(cudaMemcpy(dev_vec, vec, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec cudaMemcpy to device");

	int sum;
	int *dev_sum;

	cuda_err(cudaMalloc((void **)&dev_sum, 1 * sizeof(int)),
			 "sum cudaMalloc");

	int grid_dim = (vec_dim + BLOCK_DIM) / BLOCK_DIM;

	sum_vec<<<grid_dim, BLOCK_DIM>>>(dev_vec, dev_part_vec, vec_dim);

	// debug printing
	// int *temp = (int *)malloc(dev_part_dim * sizeof(int));
	// cuda_err(cudaMemcpy(temp, dev_part_vec, dev_part_dim * sizeof(int), cudaMemcpyDeviceToHost),
	// 		 "sum_part cudaMemcpy to host");
	// print_vec(temp, dev_part_dim);
	// debug done

	sum_vec<<<1, dev_part_dim>>>(dev_part_vec, dev_sum, dev_part_dim);

	cuda_err(cudaMemcpy(&sum, dev_sum, 1 * sizeof(int), cudaMemcpyDeviceToHost),
			 "sum cudaMemcpy to host");

	cuda_err(cudaMemcpy(vec, dev_vec, vec_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "vec cudaMemcpy to host");

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
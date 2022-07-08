#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cuda_help.h"

#define BLOCK_DIM 32

__global__ void add_matrices(int *mat_a, int *mat_b, int *mat_c, int *avs, int mat_dim)
{

	__shared__ int sh_res[BLOCK_DIM];

	int gl_ind = blockDim.x * blockIdx.x + threadIdx.x;

	if (gl_ind < mat_dim * mat_dim)
	{

		int value = mat_a[gl_ind] + mat_b[gl_ind];
		mat_c[gl_ind] = value;

		sh_res[threadIdx.x] = gl_ind;
	}

	__syncthreads();

	int my_row = gl_ind / mat_dim;

	

	return;
}

int main(void)
{

	int mat_dim = 100;

	int *mat_a = (int *)malloc(mat_dim * mat_dim * sizeof(int));
	int *mat_b = (int *)malloc(mat_dim * mat_dim * sizeof(int));
	int *mat_c = (int *)malloc(mat_dim * mat_dim * sizeof(int));
	int *row_avs = (int *)malloc(mat_dim * sizeof(int));

	init_vec(mat_a, mat_dim * mat_dim, 1);
	init_vec(mat_b, mat_dim * mat_dim, 1);
	init_vec(mat_c, mat_dim * mat_dim, 0);
	init_vec(row_avs, mat_dim, 0);

	int *dev_mat_a;
	cuda_err(cudaMalloc((void **)&dev_mat_a, mat_dim * mat_dim * sizeof(int)),
			 "mat_a cudaMalloc");

	int *dev_mat_b;
	cuda_err(cudaMalloc((void **)&dev_mat_b, mat_dim * mat_dim * sizeof(int)),
			 "mat_b cudaMalloc");

	int *dev_mat_c;
	cuda_err(cudaMalloc((void **)&dev_mat_c, mat_dim * mat_dim * sizeof(int)),
			 "mat_c cudaMalloc");

	int *dev_row_avs;
	cuda_err(cudaMalloc((void **)&dev_row_avs, mat_dim * sizeof(int)),
			 "dev_row_aws cudaMalloc");

	cuda_err(cudaMemcpy(dev_mat_a, mat_a, mat_dim * mat_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "mat_a cudaMemcpy");

	cuda_err(cudaMemcpy(dev_mat_b, mat_b, mat_dim * mat_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "mat_b cudaMemcpy");

	int grid_dim = (mat_dim * mat_dim + BLOCK_DIM) / BLOCK_DIM;

	add_matrices<<<grid_dim, BLOCK_DIM>>>(dev_mat_a, dev_mat_b, dev_mat_c, dev_row_avs, mat_dim);

	cuda_err(cudaMemcpy(mat_c, dev_mat_c, mat_dim * mat_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "mat_c cudaMemcpy");
	cuda_err(cudaMemcpy(row_avs, dev_row_avs, mat_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "row_avs cudaMemcpy");

	free(mat_a);
	free(mat_b);

	cudaFree(dev_mat_a);
	cudaFree(dev_mat_b);
	cudaFree(dev_mat_c);
	cudaFree(dev_row_avs);

	cudaDeviceReset();

	for (int i = 0; i < mat_dim * mat_dim; i++)
	{
		if (mat_c[i] != 2)
		{
			printf("Failed mat at: [%d][%d] req: %d got: %d \n",
				   i / mat_dim, i % mat_dim, 2, mat_c[i]);

			free(mat_c);
			return 1;
		}
	}
	free(mat_c);

	printf("Matrix success ... \n");

	for (int i = 0; i < mat_dim; i++)
	{
		if (row_avs[i] != 1)
		{
			printf("Failed avs at: %d ... \n", i);

			free(row_avs);
			return 1;
		}
	}
	free(row_avs);
	printf("Averages success ... \n");

	return 0;
}
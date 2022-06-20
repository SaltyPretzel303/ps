#include <stdio.h>
#include <stdlib.h>

#include "cuda_help.h"

// single warp is 32
// max thread per block are 1024 = 32x32
#define BLOCK_DIM 32

__global__ void multiply(int *mat_a, int *mat_b, int *mat_c, int dim)
{

	__shared__ int sh_a[BLOCK_DIM][BLOCK_DIM];
	__shared__ int sh_b[BLOCK_DIM][BLOCK_DIM];

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * BLOCK_DIM + ty;
	int col = bx * BLOCK_DIM + tx;

	int part_val = 0;

	for (int i = 0; i < dim / BLOCK_DIM; i++)
	{
		sh_a[ty][tx] = mat_a[row * dim + (i * BLOCK_DIM + tx)];
		sh_b[ty][tx] = mat_b[(ty + i * BLOCK_DIM) * dim + col];

		__syncthreads();

		for (int j = 0; j < BLOCK_DIM; j++)
		{
			part_val += sh_a[ty][j] * sh_b[j][tx];
		}

		__syncthreads();
	}

	mat_c[row * dim + col] = part_val;
}

int main(void)
{

	const char *in_path = "/home/nemanja/workspace/ps/cuda/src/in_mat_4.txt";
	const char *out_path = "/home/nemanja/workspace/ps/cuda/src/out_mat_4_ex.txt";

	int *mat_a;
	int *mat_b;
	int *mat_c;
	// int dim;

	int dim = 32 * 500;

	mat_a = (int *)malloc(dim * dim * sizeof(int));
	mat_b = (int *)malloc(dim * dim * sizeof(int));

	init_vec(mat_a, dim * dim);
	init_vec(mat_b, dim * dim);

	// read_mat(in_path, &mat_a, &dim);
	// read_mat(in_path, &mat_b, &dim);

	mat_c = (int *)malloc(dim * dim * sizeof(int));
	clear_vec(mat_c, dim * dim);

	int *dev_mat_a;
	int *dev_mat_b;
	int *dev_mat_c;

	cuda_err(cudaMalloc((void **)&dev_mat_a, dim * dim * sizeof(int)),
			 "mat_a cudaMalloc");

	cuda_err(cudaMalloc((void **)&dev_mat_b, dim * dim * sizeof(int)),
			 "mat_b cudaMalloc");

	cuda_err(cudaMalloc((void **)&dev_mat_c, dim * dim * sizeof(int)),
			 "mat_c cudaMalloc");

	cuda_err(cudaMemcpy(dev_mat_a, mat_a, dim * dim * sizeof(int), cudaMemcpyHostToDevice),
			 "mat_a cudaMemcpy");

	cuda_err(cudaMemcpy(dev_mat_b, mat_b, dim * dim * sizeof(int), cudaMemcpyHostToDevice),
			 "mat_a cudaMemcpy");

	free(mat_a);
	free(mat_b);

	// dim3 bl_count((dim + BLOCK_DIM) / BLOCK_DIM, (dim + BLOCK_DIM) / BLOCK_DIM);
	dim3 bl_count(dim / BLOCK_DIM, dim / BLOCK_DIM);
	dim3 bl_size(BLOCK_DIM, BLOCK_DIM);

	printf("Calculation started ... \n");
	multiply<<<bl_count, bl_size>>>(dev_mat_a, dev_mat_b, dev_mat_c, dim);
	printf("Calculation done ... \n");

	cuda_err(cudaMemcpy(mat_c, dev_mat_c, dim * dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "final cudaMemcpy");

	// program my hang here if tehre is no enough ram (actually possible situation)
	// write_mat(out_path, mat_c, dim);

	free(mat_c);

	cudaFree(dev_mat_a);
	cudaFree(dev_mat_b);
	cudaFree(dev_mat_c);

	return 0;
}
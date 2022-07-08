#include <stdio.h>
#include "cuda_help.h"

#define BLOCK_DIM 32

__host__ void seq_transpose(int *mat_a, int *mat_b, int r_dim, int c_dim)
{

	for (int i = 0; i < r_dim; i++)
	{
		for (int j = 0; j < c_dim; j++)
		{
			int a_ind = i * c_dim + j;
			int b_ind = j * r_dim + i;

			mat_b[b_ind] = mat_a[a_ind];
		}
	}
}

__global__ void transpose(int *mat_a, int *mat_b, int r_dim, int c_dim)
{

	__shared__ int sh_a[BLOCK_DIM][BLOCK_DIM + 1];

	int x_ind = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int y_ind = blockIdx.y * BLOCK_DIM + threadIdx.y;

	if (x_ind < c_dim && y_ind < r_dim)
	{
		int a_ind = y_ind * c_dim + x_ind;
		sh_a[threadIdx.y][threadIdx.x] = mat_a[a_ind];
	}

	__syncthreads();

	if (x_ind < c_dim && y_ind < r_dim)
	{
		int b_ind = x_ind * r_dim + y_ind;
		mat_b[b_ind] = sh_a[threadIdx.y][threadIdx.x];
	}
}

int main(void)
{

	int r_dim = 100;
	int c_dim = 120;

	int *in_mat = (int *)malloc(r_dim * c_dim * sizeof(int));
	int *out_mat = (int *)malloc(r_dim * c_dim * sizeof(int));

	init_vec(in_mat, r_dim * c_dim);
	clear_vec(out_mat, r_dim * c_dim);

	int *dev_in_mat;
	cuda_err(cudaMalloc((void **)&dev_in_mat, r_dim * c_dim * sizeof(int)),
			 "in_mat cudaMalloc");
	cuda_err(cudaMemcpy(dev_in_mat, in_mat, r_dim * c_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "in_mat cudaMemcpy");

	int *dev_out_mat;
	cuda_err(cudaMalloc((void **)&dev_out_mat, r_dim * c_dim * sizeof(int)),
			 "out_mat cudaMalloc");

	dim3 grid_dim((c_dim + BLOCK_DIM) / BLOCK_DIM, (r_dim + BLOCK_DIM) / BLOCK_DIM);
	dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

	transpose<<<grid_dim, block_dim>>>(dev_in_mat, dev_out_mat, r_dim, c_dim);

	cuda_err(cudaMemcpy(out_mat, dev_out_mat, r_dim * c_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "final cudaMemcpy");

	int *ctrl_vec = (int *)malloc(r_dim * c_dim * sizeof(int));
	seq_transpose(in_mat, ctrl_vec, r_dim, c_dim);

	print_vec(ctrl_vec, 2 * r_dim);
	printf("\n");

	// print_vec(out_mat, 2 * r_dim);

	for (int i = 0; i < r_dim * c_dim; i++)
	{
		if (out_mat[i] != ctrl_vec[i])
		{
			printf("Failed at ind: %d ... \n", i);

			cudaDeviceReset();
			return 1;
		}
	}

	printf("Sucess ... \n");

	cudaDeviceReset();

	return 0;
}
#include <stdio.h>
#include "cuda_help.h"

__device__ int get_row(int index, int dim)
{
	return index / dim;
}

__device__ int get_col(int index, int dim)
{
	return index % dim;
}

__device__ int get_index(int row, int col, int dim)
{
	return row * dim + col;
}

__global__ void transpose(int *in_mat, int *out_mat, int mat_dim)
{
	int gl_id = blockIdx.x * blockDim.x + threadIdx.x;

	while (gl_id < mat_dim * mat_dim)
	{
		int row = get_row(gl_id, mat_dim);
		int col = get_col(gl_id, mat_dim);

		int new_ind = get_index(col, row, mat_dim);

		out_mat[new_ind] = in_mat[gl_id];

		gl_id += gridDim.x * blockDim.x;
	}
}

int main(void)
{

	cudaEvent_t start_event;
	cudaEvent_t stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	int block_cnt = 1000;
	int block_size = 1000;

	int mat_dim = 10000;

	int *in_mat = (int *)malloc(mat_dim * mat_dim * sizeof(int));
	int *out_mat = (int *)malloc(mat_dim * mat_dim * sizeof(int));

	init_vec(in_mat, mat_dim * mat_dim);
	clear_vec(out_mat, mat_dim * mat_dim);

	// print_as_mat(in_mat, mat_dim, mat_dim);

	int *dev_in_mat;
	cuda_err(cudaMalloc((void **)&dev_in_mat, mat_dim * mat_dim * sizeof(int)),
			 "in_mat cudaMalloc");
	cuda_err(cudaMemcpy(dev_in_mat, in_mat, mat_dim * mat_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "in_mat cudaMemcpy");

	int *dev_out_mat;
	cuda_err(cudaMalloc((void **)&dev_out_mat, mat_dim * mat_dim * sizeof(int)),
			 "out_mat cudaMalloc");

	cudaEventRecord(start_event, 0);
	transpose<<<block_cnt, block_size>>>(dev_in_mat, dev_out_mat, mat_dim);
	cudaEventRecord(stop_event, 0);

	cudaEventSynchronize(stop_event);

	cuda_err(cudaMemcpy(out_mat, dev_out_mat, mat_dim * mat_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "final cudaMemcpy");

	// print_as_mat(out_mat, mat_dim, mat_dim);

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
	double tSec = elapsed_time * 1.0e-3;

	printf("Elapsed time: %f ... \n", tSec);

	free(in_mat);
	free(out_mat);
	cudaFree(dev_in_mat);
	cudaFree(dev_out_mat);

	cudaDeviceReset();

	return 0;
}
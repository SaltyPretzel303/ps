#include <stdio.h>
#include <stdlib.h>

#include "cuda_help.h"

__host__ void init_host_arrays(int *in_vec, int in_size, int *out_vec, int out_size)
{
	in_vec = (int *)malloc(in_size * sizeof(int));
	init_vec(in_vec, in_size);

	out_vec = (int *)malloc(out_size * sizeof(int));
	clear_vec(out_vec, out_size);
}

__host__ void init_device_arrays(int *host_in_vec, int *dev_in_vec, int in_dim,
								 int *dev_out_vec, int out_dim)
{

	cuda_err(cudaMalloc((void **)&dev_in_vec, in_dim * sizeof(int)),
			 "In_array cudaMalloc ... ");

	cudaMemcpy(dev_in_vec, host_in_vec, in_dim * sizeof(int), cudaMemcpyHostToDevice);

	cuda_err(cudaMalloc((void **)&dev_out_vec, out_dim * sizeof(int)),
			 "Out_array cudaMalloc");
}

__host__ int outer_compute(int *vec, int vec_dim)
{
	int sum = 0;
	for (int i = 0; i < vec_dim; i++)
	{
		sum += vec[i];
	}
	return sum;
}

__device__ int compare(int value, int el)
{
	if (value == el)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

__global__ void device_count_el(int *in_vec, int in_dim, int *out_vec, int out_dim, int el)
{

	int global_id = blockDim.x * blockIdx.x + threadIdx.x;
	out_vec[global_id] = 0;

	int check_ind = blockDim.x * blockIdx.x + threadIdx.x;

	while (check_ind <= in_dim)
	{

		out_vec[global_id] += compare(in_vec[check_ind], el);

		check_ind += gridDim.x * blockDim.x;
	}
}

int main(void)
{

	int element = 6;
	int vec_dim = 1000000;

	int bl_count = 10;
	int bl_size = 1024;
	// 1024 is max of threads per block
	// if any value above is passed kernel code will crash (or just wont execute ... )

	// compiler will yell with the warnings if variables are not initialzied 

	int *in_vec = NULL;
	int *out_vec = NULL;

	int *dev_in_vec = NULL;
	int *dev_out_vec = NULL;

	int out_dim = bl_count * bl_size; // one element for each thread

	in_vec = (int *)malloc(vec_dim * sizeof(int));
	init_vec(in_vec, vec_dim);
	in_vec[100] = element;
	in_vec[102] = element;
	in_vec[10001] = element;

	out_vec = (int *)malloc(out_dim * sizeof(int));
	clear_vec(out_vec, out_dim);

	cuda_err(cudaMalloc((void **)&dev_in_vec, vec_dim * sizeof(int)),
			 "in_cuda_malloc");

	cuda_err(cudaMalloc((void **)&dev_out_vec, out_dim * sizeof(int)),
			 "out_cuda_malloc");

	cuda_err(cudaMemcpy(dev_in_vec, in_vec, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "in_cuda_memcpy");

	printf("Doing calculation ... \n");

	device_count_el<<<bl_count, bl_size>>>(dev_in_vec, vec_dim,
										   dev_out_vec, out_dim,
										   element);

	cuda_err(cudaMemcpy(out_vec, dev_out_vec, out_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "Final cudaMemcpy ... ");

	int final_res = outer_compute(out_vec, out_dim);

	printf("Final result: %d \n", final_res);

	return 0;
}
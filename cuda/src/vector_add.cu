#include <stdio.h>
#include <stdlib.h>

#include "cuda_help.h"

void sequential_add(int *vec_a, int *vec_b, int *vec_c, int vec_dim)
{
	for (int k = 0; k < vec_dim; k++)
	{
		for (int i = 0; i < vec_dim; i++)
		{
			vec_c[i] = vec_a[i] + vec_b[i];
		}
	}
}

__global__ void parallel_add(int *vec_a, int *vec_b, int *vec_c, int vec_dim)
{

	int my_id = blockDim.x * blockIdx.x + threadIdx.x;

	while (my_id < vec_dim)
	{
		vec_c[my_id] = vec_a[my_id] + vec_b[my_id];

		my_id += blockDim.x * gridDim.x;
	}
}

int main(void)
{

	int vec_dim = 100000000;

	int *vec_a = (int *)malloc(vec_dim * sizeof(int));
	int *vec_b = (int *)malloc(vec_dim * sizeof(int));
	int *vec_c = (int *)malloc(vec_dim * sizeof(int));

	int *dev_vec_a;
	int *dev_vec_b;
	int *dev_vec_c;

	cuda_err(cudaMalloc((void **)&dev_vec_a, vec_dim * sizeof(int)),
			 "dev_vec_a alloc");

	cuda_err(cudaMalloc((void **)&dev_vec_b, vec_dim * sizeof(int)),
			 "dev_vec_b alloc");

	cuda_err(cudaMalloc((void **)&dev_vec_c, vec_dim * sizeof(int)),
			 "dev_vec_c alloc");

	init_vec(vec_a, vec_dim);
	init_vec(vec_b, vec_dim);
	clear_vec(vec_c, vec_dim);

	printf("Before: \n");
	// print_vec(vec_a, vec_dim);
	// print_vec(vec_b, vec_dim);
	// print_vec(vec_c, vec_dim);

	cuda_err(cudaMemcpy(dev_vec_a, vec_a, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec_a cpy");

	cuda_err(cudaMemcpy(dev_vec_b, vec_b, vec_dim * sizeof(int), cudaMemcpyHostToDevice),
			 "vec_v cpy");

	parallel_add<<<1000, 256>>>(dev_vec_a, dev_vec_b, dev_vec_c, vec_dim);

	cuda_err(cudaMemcpy(vec_c, dev_vec_c, vec_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "final_cpy");

	printf("After: \n");
	// print_vec(vec_a, vec_dim);
	// print_vec(vec_b, vec_dim);
	// print_vec(vec_c, vec_dim);

	return 0;
}
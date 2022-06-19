#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include "cuda_help.h"

__global__ void parallel_copy(int *vec_a, int *vec_b, int vec_dim)
{

	int block_dim = blockDim.x;
	int block_id = blockIdx.x;
	int thread_id = threadIdx.x;

	int my_id = block_dim * block_id + thread_id;

	if (my_id < vec_dim)
	{
		vec_b[my_id] = vec_a[my_id];
	}
}

int main(void)
{

	int vec_dim = 10;

	int *vec_a;
	int *vec_b;
	int *dev_vec_a;
	int *dev_vec_b;

	vec_a = (int *)malloc(vec_dim * sizeof(int));
	vec_b = (int *)malloc(vec_dim * sizeof(int));

	init_vec(vec_a, vec_dim);
	clear_vec(vec_b, vec_dim);

	printf("Before: \n");
	print_vec(vec_a, vec_dim);
	print_vec(vec_b, vec_dim);

	cuda_err(cudaMalloc((void **)&dev_vec_a, vec_dim * sizeof(int)));
	cuda_err(cudaMalloc((void **)&dev_vec_b, vec_dim * sizeof(int)));

	cuda_err(cudaMemcpy(dev_vec_a,
						vec_a,
						vec_dim * sizeof(int),
						cudaMemcpyHostToDevice));
	cuda_err(cudaMemcpy(dev_vec_b,
						vec_b,
						vec_dim * sizeof(int),
						cudaMemcpyHostToDevice));
	parallel_copy<<<1, vec_dim>>>(dev_vec_a, dev_vec_b, vec_dim);

	cuda_err(cudaMemcpy(vec_b,
						dev_vec_b,
						vec_dim * sizeof(int),
						cudaMemcpyDeviceToHost),
			 "final copy");

	printf("After: \n");
	print_vec(vec_a, vec_dim);
	print_vec(vec_b, vec_dim);

	return 0;
}

// int main(void)
// {

// 	clock_t start, end;
// 	double cpu_time_used;

// 	int *vec_a = (int *)malloc(VEC_DIM * sizeof(int));
// 	int *vec_b = (int *)malloc(VEC_DIM * sizeof(int));
// 	int *vec_c = (int *)malloc(VEC_DIM * sizeof(int));
// 	int *check_vec = (int *)malloc(VEC_DIM * sizeof(int));

// 	init_vec(vec_a, VEC_DIM);
// 	init_vec(vec_b, VEC_DIM);
// 	clear_vec(vec_c, VEC_DIM);
// 	clear_vec(check_vec, VEC_DIM);

// 	printf("Vec dim: %d ... \n", VEC_DIM);
// 	printf("Block dim: %d ... \n", BLOCK_DIM);

// 	print_vec(vec_a, VEC_DIM);
// 	print_vec(vec_b, VEC_DIM);
// 	print_vec(vec_c, VEC_DIM);

// 	int *dev_vec_a;
// 	int *dev_vec_b;
// 	int *dev_vec_c;

// 	start = clock();

// 	cudaMalloc((void **)&dev_vec_a, VEC_DIM * sizeof(int));
// 	cudaMalloc((void **)&dev_vec_b, VEC_DIM * sizeof(int));
// 	cudaMalloc((void **)&dev_vec_c, VEC_DIM * sizeof(int));

// 	cudaMemcpy(dev_vec_a, vec_a, VEC_DIM, cudaMemcpyHostToDevice);
// 	cudaMemcpy(dev_vec_b, vec_b, VEC_DIM, cudaMemcpyHostToDevice);

// 	// parallel_add<<<VEC_DIM / BLOCK_DIM, BLOCK_DIM>>>(
// 	// 	dev_vec_a, dev_vec_b, dev_vec_c, VEC_DIM);
// 	parallel_add<<<1, VEC_DIM>>>(
// 		dev_vec_a, dev_vec_b, dev_vec_c, VEC_DIM);

// 	cudaMemcpy(vec_c, dev_vec_c, VEC_DIM * sizeof(int), cudaMemcpyDeviceToHost);

// 	print_vec(vec_c, VEC_DIM);

// 	end = clock();

// 	cudaFree(dev_vec_a);
// 	cudaFree(dev_vec_b);
// 	cudaFree(dev_vec_c);

// 	sequential_add(vec_a, vec_b, check_vec, VEC_DIM);

// 	cpu_time_used = ((double)(end - start) / CLOCKS_PER_SEC);

// 	int match_result = match(vec_c, check_vec, VEC_DIM);
// 	if (match_result != 1)
// 	{

// 		printf("Parallel processing failed ... \n");
// 	}
// 	else
// 	{
// 		printf("Parallel processing successful  ... \n");
// 	}

// 	printf("With the time: %f \n", cpu_time_used);

// 	free(vec_a);
// 	free(vec_b);
// 	free(vec_c);
// 	free(check_vec);

// 	return 0;
// }

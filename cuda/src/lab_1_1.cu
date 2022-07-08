#include <stdio.h>
#include <stdlib.h>

#include "cuda_help.h"

// Koristeći CUDA tehnologiju, napisati program koji za dati niz A[n+2] računa niz B[n] na sledeći način:
// B[i] = (3*A[i] + 10* A[i+1] + 7*A[i+2]) / 20.f
// Ilustracija rada programa za n = 5 data je na slici.
// Veličinu niza A unosi korisnik. Maksimalno redukovati broj pristupa globalnoj memoriji.
// Obratiti pažnju na efikasnost paralelizacije. Omogućiti rad programa za
// nizove proizvoljne veličine.

#define BLOCK_DIM 32

__global__ void kernel(int *vec_a, int *vec_b, int vec_dim)
{

	return;

	__shared__ int sh_vec_a[BLOCK_DIM];

	// the last two thread will just move data
	// but wont calculate any value

	int grid_dim = gridDim.x * blockDim.x;

	int step = 0;

	while (step < (vec_dim + grid_dim) / grid_dim)
	{
		int gl_ind = (grid_dim * step) + blockDim.x * blockIdx.x + threadIdx.x;
		int vec_a_ind = gl_ind - blockIdx.x * 2;

		if (vec_a_ind < vec_dim)
		{
			sh_vec_a[threadIdx.x] = vec_a[vec_a_ind];
		}

		__syncthreads();

		if (threadIdx.x < BLOCK_DIM - 2) // NOTE should be <=
		{
			int i = threadIdx.x;
			int value = 3 * sh_vec_a[i] + 10 * sh_vec_a[i + 1] + 7 * sh_vec_a[i + 2] / 20.f;
			// vec_b[vec_a_ind] = value;
		}
	}
}

int main(void)
{

	return 1;

	int vec_dim = 1000;

	int *vec_a = (int *)malloc((vec_dim + 2) * sizeof(int));
	init_vec(vec_a, vec_dim, 1);
	int *vec_b = (int *)malloc(vec_dim * sizeof(int));
	init_vec(vec_b, vec_dim, 0);

	int *dev_vec_a;
	int *dev_vec_b;

	cuda_err(cudaMalloc((void **)&dev_vec_a, (vec_dim + 2) * sizeof(int)),
			 "vec_a cudaMalloc");
	cuda_err(cudaMalloc((void **)&dev_vec_b, vec_dim * sizeof(int)),
			 "vec_b cudaMalloc");

	cuda_err(cudaMemcpy(dev_vec_a, vec_a, (vec_dim + 2) * sizeof(int), cudaMemcpyHostToDevice),
			 "vec_a cudaMemcpy to device");

	int grid_dim = (vec_dim + BLOCK_DIM) / BLOCK_DIM;

	kernel<<<grid_dim, BLOCK_DIM>>>(dev_vec_a, dev_vec_b, vec_dim);

	cuda_err(cudaMemcpy(vec_b, dev_vec_b, vec_dim * sizeof(int), cudaMemcpyDeviceToHost),
			 "res_vec cudaMemcpy to host");

	cudaFree(dev_vec_a);
	cudaFree(dev_vec_b);

	cudaDeviceReset();

	int *control_vec = (int *)malloc(vec_dim * sizeof(int));
	init_vec(control_vec, vec_dim, 1);

	for (int i = 0; i < vec_dim; i++)
	{
		if (vec_b[i] != control_vec[i])
		{
			printf("Failed at: %d, req: %d is: %d ... \n",
				   i, control_vec[i], vec_b[i]);

			free(vec_b);
			return 1;
		}
	}

	printf("Success ... \n");

	free(vec_a);
	free(vec_b);

	return 0;
}

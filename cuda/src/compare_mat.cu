#include <stdio.h>
#include <stdlib.h>
#include "cuda_help.h"

int main(void)
{

	const char *in_a = "/home/nemanja/workspace/ps/cuda/src/in_mat_10.txt";
	const char *in_b = "/home/nemanja/workspace/ps/cuda/src/out_mat_10_gpu.txt";

	int *mat_a;
	int dim_a;
	read_mat(in_a, &mat_a, &dim_a);

	int *mat_b;
	int dim_b;
	read_mat(in_b, &mat_b, &dim_b);

	if (dim_a != dim_b)
	{
		printf("Dimensions are not the same ... \n");
		return 1;
	}

	for (int i = 0; i < dim_a * dim_a; i++)
	{
		if (mat_a[i] != mat_b[i])
		{
			printf("Compare failed at ind: %d -> r: %d c: %d  %d != %d ... \n",
				   i, i / dim_a, i % dim_a, mat_a[i], mat_b[i]);

			return 1;
		}
	}

	printf("Compare successful ... \n");

	return 0;
}
#include <stdio.h>
#include <stdlib.h>

void seq_multiply(int *mat_a, int *mat_b, int *mat_c, int dim)
{

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			for (int k = 0; k < dim; k++)
			{

				int a_ind = i * dim + k;
				int b_ind = k * dim + j;

				int c_ind = i * dim + j;

				mat_c[c_ind] += mat_a[a_ind] * mat_b[b_ind];
			}
		}
	}
}

int main(void)
{

	int dim = 4000;

	int *mat_a = (int *)malloc(dim * dim * sizeof(int));
	int *mat_b = (int *)malloc(dim * dim * sizeof(int));
	int *mat_c = (int *)malloc(dim * dim * sizeof(int));

	for (int i = 0; i < dim * dim; i++)
	{
		mat_a[i] = i;
		mat_b[i] = i;
		mat_c[i] = 0;
	}

	printf("Starting mult ... \n");
	seq_multiply(mat_a, mat_b, mat_c, dim);
	printf("Mult done ... \n");

	return 0;
}

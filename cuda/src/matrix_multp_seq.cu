#include <stdio.h>
#include <stdlib.h>

#include "cuda_help.h"

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

	int dims_cnt = 3;

	const char *in_path_10 = "/home/nemanja/workspace/ps/cuda/src/in_mat_10.txt";
	const char *in_path_100 = "/home/nemanja/workspace/ps/cuda/src/in_mat_100.txt";
	const char *in_path_1k = "/home/nemanja/workspace/ps/cuda/src/in_mat_1k.txt";
	const char *in_path_5k = "/home/nemanja/workspace/ps/cuda/src/in_mat_5k.txt";
	const char *in_path_10k = "/home/nemanja/workspace/ps/cuda/src/in_mat_10k.txt";

	const char *out_path_10 = "/home/nemanja/workspace/ps/cuda/src/out_mat_10.txt";
	const char *out_path_100 = "/home/nemanja/workspace/ps/cuda/src/out_mat_100.txt";
	const char *out_path_1k = "/home/nemanja/workspace/ps/cuda/src/out_mat_1k.txt";
	const char *out_path_5k = "/home/nemanja/workspace/ps/cuda/src/out_mat_5k.txt";
	const char *out_path_10k = "/home/nemanja/workspace/ps/cuda/src/out_mat_10k.txt";

	mat_desc matrices[5];
	matrices[0].path = in_path_10;
	matrices[0].dim = 10;
	matrices[0].out_path = out_path_10;

	matrices[1].path = in_path_100;
	matrices[1].dim = 100;
	matrices[1].out_path = out_path_100;

	matrices[2].path = in_path_1k;
	matrices[2].dim = 1000;
	matrices[2].out_path = out_path_1k;

	matrices[3].path = in_path_5k;
	matrices[3].dim = 5000;
	matrices[3].out_path = out_path_5k;

	matrices[4].path = in_path_10k;
	matrices[4].dim = 10000;
	matrices[5].out_path = out_path_10k;

	for (int mat = 0; mat < dims_cnt; mat++)
	{
		int dim = matrices[mat].dim;
		const char *path = matrices[mat].path;
		const char *out_path = matrices[mat].out_path;

		int *mat_a = (int *)malloc(dim * dim * sizeof(int));
		int *mat_b = (int *)malloc(dim * dim * sizeof(int));
		int *mat_c = (int *)malloc(dim * dim * sizeof(int));

		read_mat(path, &mat_a, &dim);
		read_mat(path, &mat_b, &dim);

		mat_c = (int *)malloc(dim * dim * sizeof(int));
		clear_vec(mat_c, dim * dim);

		printf("Doing: %d, outputing to: %s ... \n", dim, out_path);

		seq_multiply(mat_a, mat_b, mat_c, dim);

		write_mat(out_path, mat_c, dim);

		free(mat_a);
		free(mat_b);
		free(mat_c);

		printf("Done with: %d ... \n\n", dim);
	}

	return 0;
}
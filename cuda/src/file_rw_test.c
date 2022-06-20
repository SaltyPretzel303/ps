#include <stdio.h>
#include <stdlib.h>

void init_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = i;
	}
}

void print_vec(int *vec, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		printf("%d | ", vec[i]);
	}
	printf("\n");
}

int read_mat(const char *path, int **mat, int *dim)
{

	*mat = NULL;
	*dim = 0;

	FILE *fp;
	fp = fopen(path, "r");
	if (!fp)
	{
		printf("Failed open %s for reading ...  \n", path);
		return 1;
	}

	fscanf(fp, "%d", dim);

	int vec_dim = (*dim) * (*dim);
	*mat = (int *)malloc(vec_dim * sizeof(int));

	for (int i = 0; i < vec_dim; i++)
	{
		fscanf(fp, "%d", &(*mat)[i]);
	}

	fclose(fp);

	return 0;
}

int write_mat(const char *path, int *mat, int dim)
{
	FILE *fp;
	fp = fopen(path, "w");

	if (!fp)
	{
		printf("Failed to open %s for writing ... \n", path);
	}

	fprintf(fp, "%d\n", dim);
	for (int i = 0; i < dim * dim; i++)
	{
		fprintf(fp, "%d\n", mat[i]);
	}

	fclose(fp);
}

struct mat_desc
{
	const char *path;
	int dim;
} typedef mat_desc;

int main(void)
{

	// ATTENTION the last one with the 1M elements is not included
	// one with the 100k already has over 5GB ... (it got excluded as well)
	// the last one that is gona be written is 10k
	int dims_cnt = 5;
	// for some reasone on with the 32 doesnt work ... so the max is 5

	const char *out_path_10 = "/home/nemanja/workspace/ps/cuda/src/in_mat_10.txt";
	const char *out_path_100 = "/home/nemanja/workspace/ps/cuda/src/in_mat_100.txt";
	const char *out_path_1k = "/home/nemanja/workspace/ps/cuda/src/in_mat_1k.txt";
	const char *out_path_5k = "/home/nemanja/workspace/ps/cuda/src/in_mat_5k.txt";
	const char *out_path_4 = "/home/nemanja/workspace/ps/cuda/src/in_mat_4.txt";
	const char *out_path_32x100 = "/home/nemanja/workspace/ps/cuda/src/in_mat_32x100.txt";
	// const char *out_path_10k = "/home/nemanja/workspace/ps/cuda/src/in_mat_10k.txt";
	// const char *out_path_1M = "/home/nemanja/workspace/ps/cuda/src/in_mat_1M.txt";

	mat_desc matrices[5];
	matrices[0].path = out_path_10;
	matrices[0].dim = 10;

	matrices[1].path = out_path_100;
	matrices[1].dim = 100;

	matrices[2].path = out_path_1k;
	matrices[2].dim = 1000;

	matrices[3].path = out_path_5k;
	matrices[3].dim = 5000;

	matrices[4].path = out_path_4;
	matrices[4].dim = 4;

	// matrices[5].path = out_path_32x100;
	// matrices[5].dim = 3200;

	// matrices[4].path = out_path_10k;
	// matrices[4].dim = 10000;

	// matrices[5].path = out_path_1M;
	// matrices[5].dim = 1000000;

	for (int i = 0; i < dims_cnt; i++)
	{
		int dim = matrices[i].dim;
		const char *path = matrices[i].path;

		printf("Doing %s ... \n", path);

		int *mat = (int *)malloc(dim * dim * sizeof(int));
		init_vec(mat, dim * dim);

		FILE *fp = fopen(path, "w");

		fprintf(fp, "%d\n", dim);

		for (int i = 0; i < dim * dim; i++)
		{
			fprintf(fp, "%d\n", mat[i]);
		}

		printf("Done \t%s ... \n", path);

		fclose(fp);
	}

	return 0;
}
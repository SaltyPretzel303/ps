void cuda_err(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("We got an error from cuda: %d \n", error);
	}
}

void cuda_err(cudaError_t error, const char *at_action)
{
	if (error != cudaSuccess)
	{
		printf("We got an error from cuda: %d \n", error);
		printf("At the action: %s\n", at_action);
	}
}

void clear_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = 0;
	}
}

void init_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = i;
	}
}

int match(int *vec_a, int *vec_b, int vec_dim)
{
	for (int i = 0; i < vec_dim; i++)
	{
		if (vec_a[i] != vec_b[i])
		{
			printf("Fail at index: %d \n", i);
			return 0;
		}
	}

	return 1;
}

void print_3_num(int num)
{
	if (num < 10)
	{
		printf("0");
	}

	if (num < 100)
	{
		printf("0");
	}
	printf("%d", num);
}

void print_vec(int *vec, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		printf("%d | ", vec[i]);
	}
	printf("\n");
}

void print_as_mat(int *vec, int row_dim, int col_dim)
{
	for (int i = 0; i < row_dim * col_dim; i++)
	{
		if (i % col_dim == 0)
		{
			printf("\n");
		}

		print_3_num(vec[i]);
		printf(" | ");
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
		return 1;
	}

	fprintf(fp, "%d\n", dim);
	for (int i = 0; i < dim * dim; i++)
	{
		fprintf(fp, "%d\n", mat[i]);
	}

	fclose(fp);
	return 0;
}

struct mat_desc
{
	const char *path;
	int dim;
	const char *out_path;
} typedef mat_desc;
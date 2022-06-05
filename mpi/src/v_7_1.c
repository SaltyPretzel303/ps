#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// scatters columns of the src_mat

void init_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

void init_mat(int *mat, int dim, int value)
{
	for (int i = 0; i < dim * dim; i++)
	{
		if (value != -1)
		{
			mat[i] = value;
		}
		else
		{
			mat[i] = i;
		}
	}
}

void print_3_number(int num)
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

void print_mat(int *mat, int dim)
{
	for (int i = 0; i < dim * dim; i++)
	{

		if (i % dim == 0 && i != 0)
		{
			printf("\n");
		}

		print_3_number(mat[i]);
		printf(" | ");
	}
}

void print_vec(int *vec, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		print_3_number(vec[i]);
		printf(" | ");
	}
}

void clear_vec(int *vec, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		vec[i] = 0;
	}
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int BUFF_SIZE = 20;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int SLEEP_PERIOD = 3000;

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int MAT_DIM = 10;
	int *src_mat = (int *)malloc(MAT_DIM * MAT_DIM * sizeof(int));

	// matrix init
	if (my_rank == MASTER_RANK)
	{
		init_mat(src_mat, MAT_DIM, -1);
		print_mat(src_mat, MAT_DIM);

		printf("\n");
		printf("\n");
	}

	int bl_count = MAT_DIM;
	int bl_size = 1;
	int bl_stride = MAT_DIM;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype new_type;
	MPI_Datatype new_r_type;
	MPI_Type_vector(bl_count, bl_size, bl_stride, old_type, &new_type);
	MPI_Type_commit(&new_type);
	MPI_Type_create_resized(new_type, 0, 1 * sizeof(int), &new_r_type);
	MPI_Type_commit(&new_r_type);

	int *dest_vec = (int *)malloc(MAT_DIM * sizeof(int));
	clear_vec(dest_vec, MAT_DIM);

	MPI_Scatter(src_mat, 1, new_r_type,
				dest_vec, MAT_DIM, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(SLEEP_PERIOD * 2 * my_rank);
	printf("{%d} dest_vec: ", my_rank);
	print_vec(dest_vec, MAT_DIM);
	printf("\n");

	free(src_mat);
	free(dest_vec);

	MPI_Finalize();

	return 0;
}
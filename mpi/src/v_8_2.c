#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// scatter n/p columns (n = mat size, p = process count)
// implemented using custom type that "contains" n/p columns

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

	int MAT_DIM = 6;
	int *src_mat = (int *)malloc(MAT_DIM * MAT_DIM * sizeof(int));

	// matrix init
	if (my_rank == MASTER_RANK)
	{
		init_mat(src_mat, MAT_DIM, -1);
		print_mat(src_mat, MAT_DIM);

		printf("\n");
		printf("\n");
	}

	int n = MAT_DIM;
	int p = comm_size;

	int col_cnt = n / p;

	int bl_cnt = col_cnt * MAT_DIM;
	int bl_size = col_cnt;
	int bl_stride = MAT_DIM;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype np_col_type, np_col_r_type;

	MPI_Type_vector(bl_cnt, bl_size, bl_stride, old_type, &np_col_type);
	MPI_Type_commit(&np_col_type);
	MPI_Type_create_resized(np_col_type, 0, col_cnt * sizeof(int), &np_col_r_type);
	MPI_Type_commit(&np_col_r_type);

	int recv_cnt = MAT_DIM * col_cnt;
	int *recv_vec = (int *)malloc(recv_cnt * sizeof(int));
	clear_vec(recv_vec, recv_cnt);

	MPI_Scatter(src_mat, 1, np_col_r_type,
				recv_vec, recv_cnt, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(SLEEP_PERIOD * 2 * my_rank);
	printf("{%d} rcv_vec: ", my_rank);
	print_vec(recv_vec, recv_cnt);
	printf("\n");

	free(recv_vec);
	free(src_mat);

	MPI_Finalize();

	return 0;
}
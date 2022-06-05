#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// send elements of the upper triangular matrix from master process
// to some other process, forming the upper triangular matrix in that process as well

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

void upper_tr_print(int *mat, int dim)
{
	for (int i = 0; i < dim * dim; i++)
	{

		if (i % dim == 0 && i != 0)
		{
			printf("\n");
		}

		if (mat[i] == 0)
		{
			printf("---");
		}
		else
		{
			print_3_number(mat[i]);
		}
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

void init_upper_triangular(int *mat, int dim)
{
	int counter = 1;
	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			if (j >= i)
			{
				mat[i * dim + j] = counter;
				counter++;
			}
			else
			{
				mat[i * dim + j] = 0;
			}
		}
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

	MPI_Datatype upper_t_type;

	int bl_cnt = MAT_DIM;
	int bl_lens[bl_cnt];
	int bl_disps[bl_cnt];

	for (int i = 0; i < MAT_DIM; i++)
	{
		bl_lens[i] = MAT_DIM - i;
		bl_disps[i] = MAT_DIM + 1;
	}

	MPI_Type_indexed(bl_cnt, bl_lens, bl_disps, MPI_INTEGER, &upper_t_type);
	MPI_Type_commit(&upper_t_type);

	if (my_rank == MASTER_RANK)
	{
		int *src_mat = (int *)malloc(MAT_DIM * MAT_DIM * sizeof(int));
		init_upper_triangular(src_mat, MAT_DIM);
		upper_tr_print(src_mat, MAT_DIM);
		printf("\n\n");
		MPI_Send(src_mat, 1, upper_t_type, 1, 0, MPI_COMM_WORLD);
		free(src_mat);
	}
	else
	{
		int *dest_mat = (int *)malloc(MAT_DIM * MAT_DIM * sizeof(int));
		clear_vec(dest_mat, MAT_DIM * MAT_DIM);

		MPI_Status recv_stat;
		MPI_Recv(dest_mat, MAT_DIM * MAT_DIM, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &recv_stat);

		usleep(SLEEP_PERIOD * my_rank);
		upper_tr_print(dest_mat, MAT_DIM);
		printf("\n\n");

		free(dest_mat);
	}

	MPI_Finalize();

	return 0;
}
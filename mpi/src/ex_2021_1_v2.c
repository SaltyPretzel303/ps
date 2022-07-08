#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>
#include <math.h>

#define TRUE 1
#define FALSE 0

#define MASTER_RANK 0
#define MPI_NO_ERROR 0
#define SLEEP_PERIOD 3000

void init_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = i;
	}
}

int *get_int_vec(int len)
{
	return (int *)malloc(len * sizeof(int));
}

void clear_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = 0;
	}
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

void print_vec_as_mat(int *vec, int r_dim, int c_dim, int print_zeros)
{
	for (int i = 0; i < r_dim * c_dim; i++)
	{
		if (i % r_dim == 0)
		{
			printf("\n");
		}
		if (vec[i] == 0)
		{
			if (!print_zeros)
			{
				printf("---");
			}
			else
			{
				print_3_num(vec[i]);
			}
		}
		else
		{
			print_3_num(vec[i]);
		}
		printf(" | ");
	}
	printf("\n");
}

void print_vec(int *vec, int len, int print_zeros)
{
	for (int i = 0; i < len; i++)
	{
		if (vec[i] == 0)
		{
			if (!print_zeros)
			{
				printf("---");
			}
			else
			{
				print_3_num(vec[i]);
			}
		}
		else
		{
			print_3_num(vec[i]);
		}
		printf(" | ");
	}
	printf("\n");
}

int is_master(int my_rank)
{
	return my_rank == MASTER_RANK;
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int mat_dim = 8;
	int proc_dim = sqrt(comm_size);

	int *mat = get_int_vec(mat_dim * mat_dim);
	init_vec(mat, mat_dim * mat_dim);
	int *vec = get_int_vec(mat_dim);
	init_vec(vec, mat_dim);

#pragma region scatter mat blocks

	int bl_count = mat_dim / proc_dim;
	int bl_lens = mat_dim / proc_dim;
	int bl_stride = mat_dim;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype mat_part;

	MPI_Type_vector(bl_count, bl_lens, bl_stride, old_type, &mat_part);
	MPI_Type_commit(&mat_part);

	int lb = 0;
	int extent = bl_lens * sizeof(int);
	MPI_Datatype rs_mat_part;
	MPI_Type_create_resized(mat_part, lb, extent, &rs_mat_part);
	MPI_Type_commit(&rs_mat_part);

	int rcv_count = bl_count * bl_lens;
	int *loc_mat = get_int_vec(rcv_count);
	clear_vec(loc_mat, rcv_count);

	MPI_Scatter(mat, 1, rs_mat_part,
				loc_mat, rcv_count, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(3000 * my_rank);

	printf("{%d} loc_mat: ", my_rank);
	print_vec_as_mat(loc_mat, bl_lens, bl_lens, FALSE);

#pragma endregion

	int vec_bl_size = mat_dim / proc_dim;

	int *loc_vec = get_int_vec(vec_bl_size);
	clear_vec(loc_vec, vec_bl_size);

	MPI_Scatter(vec, vec_bl_size, MPI_INTEGER,
				loc_vec, vec_bl_size, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(3000 * my_rank * 4);

	printf("{%d} loc_vec: ", my_rank);
	print_vec(loc_vec, vec_bl_size, FALSE);

	MPI_Finalize();

	return 0;
}
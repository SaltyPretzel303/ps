#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0
#define MAX_INT 999999

// Write a program for matrix multiplication of matrix A(MxN) and  B(NxK) with the result
// of matrix C. Matrix A and B are initialized in the master process. Rows of the matrix A
// are separated in to the blocks (p-number of processes divides M). Each process Pi will
// receive rows with the indices L, L mode P = i (0<=i<=p-1) example: I, I+P, I+2P ... I+M-P.
// Master process will distribute righ blocks and the whole matrix B to every process.
// Sending of the rows should be done with the single command. The result of matrix
// multiplication should be printed in the process containing the minimum of the
// matrix values.

void init_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

void clear_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = 0;
	}
}

void print_4_num(int num)
{
	if (num < 10)
	{
		printf("0");
	}
	if (num < 100)
	{
		printf("0");
	}
	if (num < 1000)
	{
		printf("0");
	}

	printf("%d", num);
}

void print_vec_as_mat(int *vec, int mat_dim_r, int mat_dim_c)
{
	for (int i = 0; i < mat_dim_r * mat_dim_c; i++)
	{
		if (i % mat_dim_c == 0)
		{
			printf("\n");
		}

		if (vec[i] == 0)
		{
			printf("----");
		}
		else
		{
			print_4_num(vec[i]);
		}
		printf(" | ");
	}
	printf("\n");
}

struct two_ints
{
	int index;
	int value;

} typedef two_ints;

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

	int same_dim = 6;

	int mat_a_r_dim = same_dim;
	int mat_a_c_dim = same_dim;
	int mat_b_r_dim = same_dim;
	int mat_b_c_dim = same_dim;

	int *mat_a;
	int *mat_b;
	int *mat_c;
	int *final_mat;

	two_ints loc_min;
	two_ints world_min;

	if (my_rank == MASTER_RANK)
	{
		mat_a = (int *)malloc(mat_a_r_dim * mat_a_c_dim * sizeof(int));
		mat_b = (int *)malloc(mat_b_r_dim * mat_b_c_dim * sizeof(int));
		init_vec(mat_a, mat_a_r_dim * mat_a_c_dim);
		init_vec(mat_b, mat_b_r_dim * mat_b_c_dim);

		printf("Matrix A: \n");
		print_vec_as_mat(mat_a, mat_a_r_dim, mat_a_c_dim);
		printf("Matrix B: \n");
		print_vec_as_mat(mat_b, mat_b_r_dim, mat_b_c_dim);
		printf("===========================================================\n");
	}

	mat_c = (int *)malloc(mat_a_r_dim * mat_b_c_dim * sizeof(int));
	clear_vec(mat_c, mat_a_r_dim * mat_b_c_dim);

	final_mat = (int *)malloc(mat_a_r_dim * mat_b_c_dim * sizeof(int));
	clear_vec(final_mat, mat_a_r_dim * mat_b_c_dim);

	int bl_cnt = mat_a_r_dim / comm_size;
	int bl_size = mat_a_c_dim;
	int bl_stride = mat_a_c_dim * comm_size;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype rows_type;
	MPI_Datatype rows_rs_type;

	MPI_Type_vector(bl_cnt, bl_size, bl_stride, old_type, &rows_type);
	MPI_Type_commit(&rows_type);
	int lb = 0;
	int extend = mat_a_c_dim * sizeof(int);
	MPI_Type_create_resized(rows_type, lb, extend, &rows_rs_type);
	MPI_Type_commit(&rows_rs_type);

	int loc_a_rows_len = mat_a_c_dim * (mat_a_r_dim / comm_size);
	int *loc_a_rows = (int *)malloc(loc_a_rows_len * sizeof(int));

	MPI_Scatter(mat_a, 1, rows_rs_type,
				loc_a_rows, loc_a_rows_len, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	MPI_Bcast(mat_b, mat_b_r_dim * mat_b_c_dim, MPI_INTEGER, MASTER_RANK, MPI_COMM_WORLD);

	loc_min.index = my_rank;
	loc_min.value = MAX_INT;

#pragma region calculate local matrix product

	int a_rows_cnt = mat_a_r_dim / comm_size;
	for (int i = 0; i < a_rows_cnt; i++)
	{
		int a_row = my_rank + i * comm_size;

		for (int j = 0; j < mat_a_c_dim; j++)
		{

			for (int k = 0; k < mat_a_c_dim; k++)
			{
				int a_value = loc_a_rows[i * mat_a_c_dim + k];
				int b_value = mat_b[k * mat_b_c_dim + j];

				if (a_value < loc_min.value)
				{
					loc_min.value = a_value;
				}
				// commnted just for testing
				// for valid result uncomment next if
				// if (b_value < loc_min.value)
				// {
				// 	loc_min.value = b_value;
				// }

				int value = a_value * b_value;

				mat_c[a_row * mat_a_c_dim + j] += value;
			}
		}
	}

#pragma endregion

#pragma region print the calculated data

	usleep(SLEEP_PERIOD * 4 * my_rank);

	// print received portion of mat_a
	for (int i = 0; i < loc_a_rows_len; i++)
	{
		if (i % mat_a_c_dim == 0)
		{
			printf("\n");
		}
		print_4_num(loc_a_rows[i]);
		printf(" | ");
	}
	printf("\n");

	// print the calculated parts of the mat_c
	print_vec_as_mat(mat_c, mat_a_r_dim, mat_b_c_dim);

	printf("===========================================================\n");

#pragma endregion

	MPI_Gather(&mat_c[my_rank * mat_a_c_dim], 1, rows_rs_type,
			   final_mat, 1, rows_rs_type,
			   MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		usleep(SLEEP_PERIOD * 4 * comm_size);
		printf("Final result in master ... \n");
		print_vec_as_mat(final_mat, mat_a_r_dim, mat_b_c_dim);
		printf("\n");
	}

	MPI_Reduce(&loc_min, &world_min, 1, MPI_2INTEGER, MPI_MINLOC, MASTER_RANK, MPI_COMM_WORLD);

	MPI_Bcast(&world_min, 1, MPI_2INTEGER, MASTER_RANK, MPI_COMM_WORLD);
	MPI_Bcast(final_mat, mat_a_r_dim * mat_b_c_dim, MPI_INTEGER, MASTER_RANK, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD); // just wait for all other prints to happen
	if (my_rank == world_min.index)
	{
		printf("{%d} Proc with ind: %d had min value of: %d\n",
			   my_rank, world_min.index, world_min.value);
		print_vec_as_mat(final_mat, mat_a_r_dim, mat_b_c_dim);
		printf("\n");
	}

#pragma region free allocated memory

	if (my_rank == MASTER_RANK)
	{
		free(mat_a);
		free(mat_b);
	}

	free(loc_a_rows);

#pragma endregion

	MPI_Finalize();

	return 0;
}
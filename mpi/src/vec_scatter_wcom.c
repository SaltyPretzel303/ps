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

	int proc_dim = sqrt(comm_size);

	int per_proc_el = 2;
	int vec_dim = proc_dim * per_proc_el;
	if (my_rank == 0)
	{
		printf("proc_dim: %d, per_proc_el: %d, vec_dim: %d \n",
			   proc_dim, per_proc_el, vec_dim);
	}
	int *vec = get_int_vec(vec_dim);
	init_vec(vec, vec_dim); // each process will get 3 elements

	int proc_row = my_rank / proc_dim;
	int proc_col = my_rank % proc_dim;

	usleep(2000 * my_rank);
	printf("{%d} row: %d, col: %d\n", my_rank, proc_row, proc_col);

	int *rec_vec = get_int_vec(per_proc_el);
	clear_vec(rec_vec, per_proc_el);

	MPI_Comm row_com;
	MPI_Comm_split(MPI_COMM_WORLD, proc_row, proc_col, &row_com);

	int in_row_rank;
	MPI_Comm_rank(row_com, &in_row_rank);

	int is_first_c = in_row_rank == 0;
	MPI_Comm first_col_comm;
	MPI_Comm_split(MPI_COMM_WORLD, is_first_c, my_rank, &first_col_comm);

	int fc_rank;
	MPI_Comm_rank(first_col_comm, &fc_rank);

	MPI_Scatter(vec, per_proc_el, MPI_INTEGER,
				rec_vec, per_proc_el, MPI_INTEGER,
				MASTER_RANK, first_col_comm);

	MPI_Bcast(rec_vec, per_proc_el, MPI_INTEGER, MASTER_RANK, row_com);

	usleep(3000 * proc_row);
	printf("{%d}: ", my_rank);
	print_vec(rec_vec, per_proc_el, TRUE);
	printf("========================\n");

	MPI_Finalize();

	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Proces 0 kreira matricu reda n i šalje i - om procesu po dve kvazidijagonale matrice, obe
// na udaljenosti i od glavne dijagonale.Proces i kreira svoju matricu tako što smešta
// primljene dijagonale u prvu i drugu kolonu matrice a ostala mesta popunjava nulama.
// Napisati MPI program koji realizuje opisanu komunikaciju korišćenjem izvedenih tipova
// podataka i prikazuje vrednosti odgovarajućih kolona.

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

	int mat_dim = 10;
	int target_proc = 1;

	int *mat;
	if (is_master(my_rank))
	{
		mat = get_int_vec(mat_dim * mat_dim);
		init_vec(mat, mat_dim * mat_dim);
		print_vec_as_mat(mat, mat_dim, mat_dim, TRUE);
	}

	int bl_cnt = mat_dim - target_proc;
	int bl_size = 1;
	int bl_stride = mat_dim - 1;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype diag_type;

	MPI_Type_vector(bl_cnt, bl_size, bl_stride, old_type, &diag_type);
	MPI_Type_commit(&diag_type);

	int *rec_mat;

	if (my_rank == target_proc)
	{
	}

	if (is_master(my_rank))
	{
		int diag_1_ind = mat_dim - 1 - target_proc;
		int diag_2_ind = (target_proc * mat_dim) + mat_dim - 1;
		MPI_Send(&mat[diag_1_ind], 1, diag_type, target_proc, 0, MPI_COMM_WORLD);
		MPI_Send(&mat[diag_2_ind], 1, diag_type, target_proc, 0, MPI_COMM_WORLD);
	}
	else if (my_rank == target_proc)
	{
		int rec_bl_cnt = mat_dim - target_proc;
		int rec_bl_size = 1;
		int rec_bl_stride = mat_dim;
		MPI_Datatype rec_old_type = MPI_INTEGER;
		MPI_Datatype rec_col_type;
		MPI_Type_vector(rec_bl_cnt, rec_bl_size, rec_bl_stride, rec_old_type, &rec_col_type);
		MPI_Type_commit(&rec_col_type);

		rec_mat = get_int_vec(mat_dim * mat_dim);
		clear_vec(rec_mat, mat_dim * mat_dim);

		MPI_Recv(&rec_mat[0], 1, rec_col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&rec_mat[1], 1, rec_col_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		usleep(comm_size * 2 * SLEEP_PERIOD);
		printf("rec mat: \n");
		print_vec_as_mat(rec_mat, mat_dim, mat_dim, FALSE);
	}

	MPI_Finalize();

	return 0;
}
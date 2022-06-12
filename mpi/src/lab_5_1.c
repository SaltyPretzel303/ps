#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Napisati MPI program koji kreira komunikator koji se sastoji od dijagonalnih procesa u
// kvadratnoj mreži procesa.Iz master procesa novog komunikatora poslati poruku svim ostalim
// procesima.Svaki proces novog komunikatora treba da prikaže primljenu poruku,
// identifikator procesa u novom komunikatoru i stari identifikator procesa.

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

	int ndims = 2;
	int dims[2] = {4, 4};
	int periods[2] = {1, 1};
	MPI_Comm cart_comm;

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_comm);

	int cart_rank = -1;
	int coords[2] = {-1, -1};
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, cart_rank, ndims, coords);

	int color = 0;
	if (coords[0] == coords[1])
	{
		color = 1;
	}

	int diag_rank = -1;
	MPI_Comm diag_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, 0, &diag_comm);
	MPI_Comm_rank(diag_comm, &diag_rank);

	int message_dim = 7;
	char message[] = {'m',
					  'e',
					  's',
					  's',
					  'a',
					  'g',
					  'e'};

	if (color == 1)
	{
		MPI_Bcast(message, message_dim, MPI_CHAR, MASTER_RANK, diag_comm);

		usleep(my_rank * SLEEP_PERIOD);
		printf("world: {%d} coords: [%d,%d] diag: %d\n", my_rank, coords[0], coords[1], diag_rank);
	}
	MPI_Finalize();

	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Napisati MPI program kojim se kreira dvodimenzionalna Cartesian struktura sa n vrsta i m
// kolona.U svakom od nxm procesa odštampati identifikatore procesa njegovog levog i desnog
// suseda na udaljenosti 2. Smatrati da su procesi u prvoj i poslednjoj koloni jedne vrste susedni.

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

	int r_dims = 4;
	int c_dims = 4;

	int ndims = 2;
	int dims[2] = {r_dims, c_dims};
	int periods[2] = {1, 1};
	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_comm);

	int cart_rank;
	int coords[2];
	MPI_Comm_rank(cart_comm, &cart_rank);
	MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

	int left_rank;
	int left_coords[2];
	int right_rank;
	int right_coords[2];
	MPI_Cart_shift(cart_comm, 1, -2, &cart_rank, &left_rank);
	MPI_Cart_shift(cart_comm, 1, 2, &cart_rank, &right_rank);

	MPI_Cart_coords(cart_comm, left_rank, 2, left_coords);
	MPI_Cart_coords(cart_comm, right_rank, 2, right_coords);

	usleep(my_rank * 2 * SLEEP_PERIOD);
	printf("world: {%d} = cart: {%d} [%d,%d]  <-[%d,%d] [%d,%d]->\n",
		   my_rank, cart_rank, coords[0], coords[1],
		   left_coords[0], left_coords[1], right_coords[0], right_coords[1]);

	MPI_Finalize();

	return 0;
}
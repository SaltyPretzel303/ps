#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

// Every process contains 30 integers. Find the maximum value on each of the
// 30 positions and the index of process that contains it.

void init_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

void print_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		printf("%d | ", buff[i]);
	}
}

struct INDEXED_VALUE
{
	int value;
	int index;
} typedef INDEXED_VALUE;

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int BUFF_SIZE = 10;
	const int VALUE_LIMIT = 1000;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int CONST_DELAY = 30000;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	INDEXED_VALUE *buff = (INDEXED_VALUE *)malloc(BUFF_SIZE * sizeof(INDEXED_VALUE));

	// sleep will cause different seed for each process
	// and prettier printing
	usleep(CONST_DELAY * my_rank);
	srand(time(0) + my_rank);

	for (int i = 0; i < BUFF_SIZE; i++)
	{
		buff[i].index = my_rank;
		buff[i].value = rand() % VALUE_LIMIT;
	}

	printf("{%d}->\t", my_rank);
	for (int i = 0; i < BUFF_SIZE; i++)
	{
		printf("%d\t ", buff[i].value = rand() % VALUE_LIMIT);
	}
	printf("\n");

	// ================================================

	INDEXED_VALUE *max_results;
	if (my_rank == MASTER_RANK)
	{
		max_results = (INDEXED_VALUE *)malloc(BUFF_SIZE * sizeof(INDEXED_VALUE));
	}

	MPI_Reduce(buff, max_results, BUFF_SIZE, MPI_2INTEGER, MPI_MAXLOC,
			   MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		// print values
		printf("{%d} Received max. values ...  \n", my_rank);
		printf("{%d}->\t", my_rank);
		for (int i = 0; i < BUFF_SIZE; i++)
		{
			printf("%d\t ", max_results[i].value);
		}
		printf("\n");

		// print indices
		printf("{%d}->\t", my_rank);
		for (int i = 0; i < BUFF_SIZE; i++)
		{
			printf("%d\t ", max_results[i].index);
		}
		printf("\n");

		free(max_results);
	}

	free(buff);

	MPI_Finalize();

	return 0;
}
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

// Find a sum of N integers using ptp communicaiton model is such way that
// every process participate in summation.

void init_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int BUFF_SIZE = 20;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int buff = my_rank;
	int sum = -1;
	MPI_Reduce(&buff, &sum, 1, MPI_INTEGER, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		printf("{%d} Sum in master rank: %d\n", my_rank, sum);
	}

	MPI_Finalize();

	return 0;
}
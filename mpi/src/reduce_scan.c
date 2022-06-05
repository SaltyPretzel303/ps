#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

// Just a simple example of MPI_Scan 

void init_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int my_value = my_rank;
	int part_sum = 0;

	MPI_Scan(&my_value, &part_sum, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

	printf("{%d} Part sum is: %d\n", my_rank, part_sum);

	MPI_Finalize();

	return 0;
}
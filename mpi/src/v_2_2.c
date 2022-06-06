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

	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int value = my_rank;
	int part_sum = 0;

	if (my_rank == MASTER_RANK)
	{
		part_sum += value;
		MPI_Send(&part_sum, 1, MPI_INTEGER, my_rank + 1, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Status recv_status;
		MPI_Recv(&part_sum, 1, MPI_INTEGER, my_rank - 1, 0, MPI_COMM_WORLD, &recv_status);
		printf("{%d} Received: %d\n", my_rank, part_sum);
		// if (recv_status.MPI_ERROR != MPI_NO_ERROR)
		// {
		// 	printf("{%d} We received mpi error from %d, error: %d\n",
		// 		   my_rank, my_rank - 1, recv_status.MPI_ERROR);
		// }

		part_sum += my_rank;
		if (my_rank < comm_size - 1)
		{
			MPI_Send(&part_sum, 1, MPI_INTEGER, my_rank + 1, 0, MPI_COMM_WORLD);
		}
		else
		{
			printf("{%d} As a last process sum is: %d\n", my_rank, part_sum);
		}
	}

	MPI_Finalize();

	return 0;
}
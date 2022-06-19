#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

// Simple scatter example to show that if the process count is greater than the
// buff size, some of the processes will simple not receive any value ...

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

	const int BUFF_SIZE = 10;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int SLEEP_PERIOD = 20000;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int *buff = (int *)malloc(BUFF_SIZE * sizeof(MPI_INTEGER));
	init_vec(buff, BUFF_SIZE);

	int local_value = -1;

	MPI_Scatter(buff, 1, MPI_INTEGER, &local_value, 1, MPI_INTEGER, MASTER_RANK, MPI_COMM_WORLD);

	free(buff);

	usleep(SLEEP_PERIOD * my_rank);
	printf("{%d} received: {%d}\n", my_rank, local_value);

	MPI_Finalize();

	return 0;
}
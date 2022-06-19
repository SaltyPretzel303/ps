#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <unistd.h>

// Just a simple example of MPI_Scatter and MPI_Gather

void init_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

void clear_buff(int *buff, int len)
{
	memset(buff, 0, len * sizeof(int));
}

void print_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		printf("%d | ", buff[i]);
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

	int *buff = (int *)malloc(comm_size * sizeof(MPI_INTEGER));

	if (my_rank == MASTER_RANK)
	{
		init_vec(buff, comm_size);
		printf("{%d} Sending buffer\n", my_rank);
		print_buff(buff, comm_size);
		printf("\n");
	}

	int local_value;
	MPI_Scatter(buff, 1, MPI_INTEGER, &local_value, 1, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	free(buff);

	printf("{%d} Received: %d\n", my_rank, local_value);

	// ============================================================

	int *recv_buff;
	recv_buff = (int *)malloc(comm_size * sizeof(MPI_INTEGER));

	sleep(1);

	MPI_Gather(&local_value, 1, MPI_INTEGER, &recv_buff[0], 1, MPI_INTEGER,
			   MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		printf("{%d} In the master process: \n", my_rank);
		print_buff(recv_buff, comm_size);
		printf("\n");
	}

	free(recv_buff);

	MPI_Finalize();

	return 0;
}
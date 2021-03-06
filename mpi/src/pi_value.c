#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

// Calculate value of number PI as a value of integral of function f(x) = 4 / (1+x^2)
// on interval [0,1].

// Napisati MPI program koji izračunava vrednost broja PI kao vrednost
// integrala funkcije f(x) = 4 / (1 + x^2) na intervalu[0, 1].

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

	if (my_rank == MASTER_RANK)
	{
		int *buff = (int *)malloc(BUFF_SIZE * sizeof(int));
		init_vec(buff, BUFF_SIZE);

		printf("{%d} Sending buffer\n", my_rank);
		MPI_Send(buff, BUFF_SIZE, MPI_INTEGER, 1, 0, MPI_COMM_WORLD);

		free(buff);
	}
	else
	{

		int *recv_buff = (int *)malloc(BUFF_SIZE * sizeof(int));
		MPI_Status recv_status;

		MPI_Recv(recv_buff, BUFF_SIZE, MPI_INTEGER, MASTER_RANK, 0, MPI_COMM_WORLD, &recv_status);
		if (recv_status.MPI_ERROR != MPI_NO_ERROR)
		{
			printf("In proc. %d we got an MPI_ERROR: %d", my_rank, recv_status.MPI_ERROR);
			return recv_status.MPI_ERROR;
		}

		for (int i = 0; i < BUFF_SIZE; i++)
		{
			printf("{%d} Buff[%d] \t=>\t %d\n", my_rank, i, recv_buff[i]);
		}

		printf("\n");

		free(recv_buff);
	}

	MPI_Finalize();

	return 0;
}
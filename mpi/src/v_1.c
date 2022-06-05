#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

// Master process (rank==0) reads integer values from the standard input.
// Process with the rank i (except the master process) receives the value
// from the process with the rank i-1 and sends it to the process witht the rank
// i+1. Processing is terminated when process receives negative value (in the master
// process that is the moment when the negative value is read and propagated).

// Napisati program koji uzima podatke od nultog procesa i
// šalje ih svim drugim procesima tako što proces i treba da primi podatke
// i pošalje ih procesu i+1,sve dok se ne stigne do poslednjeg procesa.
// Unos podataka se završava nakon što se prenese negativna vrednost podatka.

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

	const int BUFF_SIZE = 20;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int SLEEP_MULT = 800;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if (comm_size < 2)
	{
		printf("Not enough processes (minimum is 2) ... \n");
		return 0;
	}

	if (my_rank == MASTER_RANK)
	{
		int still_valid = 1;

		while (still_valid)
		{
			int buff;
			printf("Please enter the int value: ");
			scanf("%d", &buff);

			if (buff < 0)
			{
				printf("{%d} Value negative, last send ... \n", my_rank);
				still_valid = 0;
			}

			MPI_Send(&buff, 1, MPI_INTEGER, 1, 0, MPI_COMM_WORLD);
		}
	}
	else
	{

		int still_valid = 1;

		while (still_valid)
		{
			int recv_buff;
			MPI_Status recv_status;

			MPI_Recv(&recv_buff, 1, MPI_INTEGER, my_rank - 1, 0, MPI_COMM_WORLD, &recv_status);
			if (recv_status.MPI_ERROR != MPI_NO_ERROR)
			{
				printf("{%d} We got an MPI error: %d", my_rank, recv_status.MPI_ERROR);
			}
			else
			{

				if (recv_buff < 0)
				{
					printf("{%d} Received negative value, last receive ... \n", my_rank);
					still_valid = 0;
				}
				else
				{
					printf("{%d} Received positive value ... \n", my_rank);
				}

				if (my_rank < comm_size - 1)
				{
					MPI_Send(&recv_buff, 1, MPI_INTEGER, my_rank + 1, 0, MPI_COMM_WORLD);
				}
			}
		}
	}

	MPI_Finalize();

	return 0;
}
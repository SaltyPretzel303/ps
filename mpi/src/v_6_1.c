#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Napisati MPI program kojim se vrši podela procesa članova
// komunikatora MPI_COMM_WORLD u dve grupe : grupu procesa sa neparnim
// identifikatorima i grupu procesa sa parnim identifikatorima

// split MPI_COMM_WORLD in to the two communicators one containing only processes
// with the even rank (in MPI_COMM_WORLD) and the other one with the processes
// having odd rank

// implemented using MPI_Comm_split

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
	const int SLEEP_PERIOD = 3000;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int group;
	MPI_Comm_group(MPI_COMM_WORLD, &group);

	int color = my_rank % 2;

	int NEW_COMM;
	MPI_Comm_split(MPI_COMM_WORLD, color, 0, &NEW_COMM);

	int new_rank;
	MPI_Comm_rank(NEW_COMM, &new_rank);

	usleep(SLEEP_PERIOD * my_rank);

	printf("{%d} new_rank: %d \n", my_rank, new_rank);

	MPI_Finalize();

	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// split MPI_COMM_WORLD in to the two communicators (or just groups) one containing
// only processeswith the even rank (in MPI_COMM_WORLD) and the other one with
// the processes having odd rank

// implemented using MPI_Comm_include and MPI_Comm_exclude

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
	const int SLEEP_PERIOD = 3000;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	int color = my_rank % 2;

	// max possible size (if all of them are even)
	int *even_ranks = (int *)malloc(comm_size * sizeof(int));
	int even_count = 0;
	for (int i = 0; i < comm_size; i++)
	{
		if (i % 2 == 0)
		{
			even_ranks[even_count] = i;
			even_count++;
		}
	}

	int even_group, odd_group;

	MPI_Group_incl(world_group, even_count, even_ranks, &even_group);
	MPI_Group_excl(world_group, even_count, even_ranks, &odd_group);

	free(even_ranks);

	int new_even_rank, new_odd_rank;
	MPI_Group_rank(even_group, &new_even_rank);
	MPI_Group_rank(odd_group, &new_odd_rank);

	usleep(SLEEP_PERIOD * my_rank * 2);
	printf("{%d} \tnew_even_rank: %d, \tnew_odd_rank: %d \n",
		   my_rank, new_even_rank, new_odd_rank);

	MPI_Finalize();

	return 0;
}
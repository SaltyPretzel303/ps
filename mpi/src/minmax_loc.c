#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

// Just a simple example of MPI_MINLOC and MPI_MAXLOC

void init_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

struct minmax_type
{
	float len;
	int index;
};

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

	struct minmax_type value;
	value.len = my_rank * 10;
	value.index = my_rank;

	struct minmax_type max_loc;
	struct minmax_type min_loc;

	MPI_Reduce(&value.len, &max_loc, 1, MPI_FLOAT_INT, MPI_MAXLOC, MASTER_RANK, MPI_COMM_WORLD);
	MPI_Reduce(&value.len, &min_loc, 1, MPI_FLOAT_INT, MPI_MINLOC, MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		printf("{%d} MinLoc = %d with the value: %f \n",
			   my_rank, min_loc.index, min_loc.len);
		printf("{%d} MaxLoc = %d with the value: %f \n",
			   my_rank, max_loc.index, max_loc.len);
	}

	MPI_Finalize();

	return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// attempted to sactter vector across different communicator
// master process is in world comm
// workers are in workers_comm
// does not work

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

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int excl_cnt = 1;
	int excl_array[1] = {MASTER_RANK};

	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group workers_group;
	MPI_Comm workers_comm;

	MPI_Group_excl(world_group, excl_cnt, excl_array, &workers_group);
	MPI_Comm_create(MPI_COMM_WORLD, workers_group, &workers_comm);

	int array_len = 10;
	int *array = (int *)malloc(array_len * sizeof(int));
	init_vec(array, array_len);

	int recv_buff = -1;
	MPI_Scatter(array, 1, MPI_INTEGER,
				&recv_buff, 1, MPI_INTEGER,
				MASTER_RANK, workers_comm);

	int workers_rank;
	MPI_Comm_rank(workers_comm, &workers_rank);

	usleep(SLEEP_PERIOD * 2 * world_rank);
	printf("{%d} worker: %d, value: %d\n", world_rank, workers_rank, recv_buff);

	MPI_Finalize();

	return 0;
}
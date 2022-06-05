#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

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

	int bl_count = 10;
	int bl_size = 1;
	int bl_stride = 2;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype new_type;

	MPI_Type_vector(bl_count, bl_size, bl_stride, old_type, &new_type);
	MPI_Type_commit(&new_type);

	int vec_len = 20;
	int *vec = (int *)malloc(vec_len * sizeof(int));

	if (my_rank == MASTER_RANK)
	{
		init_buff(vec, vec_len);
		MPI_Send(vec, 1, new_type, 1, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Status recv_stat;
		MPI_Recv(vec, vec_len, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &recv_stat);

		for (int i = 0; i < vec_len; i++)
		{
			printf("%d | ", vec[i]);
		}
		printf("\n");
	}

	MPI_Finalize();

	return 0;
}
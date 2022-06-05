#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// master process reads one integer and one float number and then bcast them
// to the rest of the processes using single command/send_method

void init_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

struct two_values
{
	int int_value;
	double double_value;

} typedef two_values_type;

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int BUFF_SIZE = 20;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int SLEEP_PERIOD = 3000;

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	two_values_type src_value;

	MPI_Aint indices[2];
	MPI_Address(&src_value.int_value, &indices[0]);
	MPI_Address(&src_value.double_value, &indices[1]);

	int cnt = 2;
	int bl_lens[2] = {1, 1};
	MPI_Aint bl_disps[2] = {0, indices[1] - indices[0]};
	MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
	MPI_Datatype two_val_type;

	MPI_Type_create_struct(cnt, bl_lens, bl_disps, types, &two_val_type);
	MPI_Type_commit(&two_val_type);

	if (my_rank == MASTER_RANK)
	{
		// imagine that the rest of the program  is in the while loop
		// and this values are read from std input
		src_value.int_value = 100;
		src_value.double_value = 100.5213123;
	}

	MPI_Bcast(&src_value, 1, two_val_type, MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank != MASTER_RANK)
	{
		usleep(SLEEP_PERIOD * 2 * my_rank);

		printf("{%d} recv: %d %lf \n", my_rank, src_value.int_value, src_value.double_value);
	}

	MPI_Finalize();

	return 0;
}
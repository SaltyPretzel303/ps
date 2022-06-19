#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// scatter elemets of the vec. using custom type
// proc Pi will get i, i+p, i+2p ... i+dim-p; p=comm_size, dim=vector_len
// and then gather them again using the same custom type

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

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int vec_dim = 10;
	int *vec;
	if (my_rank == MASTER_RANK)
	{
		vec = (int *)malloc(vec_dim * sizeof(int));
		init_vec(vec, vec_dim);

		for (int i = 0; i < vec_dim; i++)
		{
			printf("%d, ", vec[i]);
		}
		printf("\n");
	}

	int bl_cnt = vec_dim / comm_size;
	int bl_size = 1;
	int bl_stride = comm_size;

	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype vec_type;
	MPI_Type_vector(bl_cnt, bl_size, bl_stride, old_type, &vec_type);
	MPI_Type_commit(&vec_type);
	MPI_Datatype vec_rs_type;
	int lb = 0;
	int extent = 1 * sizeof(int);
	MPI_Type_create_resized(vec_type, lb, extent, &vec_rs_type);
	MPI_Type_commit(&vec_rs_type);

	int loc_dim = vec_dim / comm_size;
	int *recv_buff = (int *)malloc(loc_dim * sizeof(int));

	MPI_Scatter(vec, 1, vec_rs_type,
				recv_buff, loc_dim, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(SLEEP_PERIOD * 2 * my_rank);
	printf("{%d} proc: ", my_rank);
	for (int i = 0; i < loc_dim; i++)
	{
		printf("%d, ", recv_buff[i]);
	}
	printf("\n");

	int *final_arr;
	if (my_rank == MASTER_RANK)
	{
		final_arr = (int *)malloc(vec_dim * sizeof(int));
		for (int i = 0; i < vec_dim; i++)
		{
			final_arr[i] = 0;
		}
	}

	MPI_Gather(recv_buff, vec_dim / comm_size, MPI_INTEGER,
			   final_arr, 1, vec_rs_type,
			   MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		usleep(SLEEP_PERIOD * comm_size * 2);
		printf("==================================\n");
		for (int i = 0; i < vec_dim; i++)
		{
			printf("%d, ", final_arr[i]);
		}
		printf("\n");
	}

	MPI_Finalize();

	return 0;
}
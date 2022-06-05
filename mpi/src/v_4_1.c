#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <unistd.h>

// Calculate scalar product of two vectors;
// !! works only if BUFF_SIZE is divideable by process count

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

	const int BUFF_SIZE = 8;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int CONST_DELAY = 20000;
	const int MAX_VEC_VALUE = 20;
	// just to remember the name of the mpi constant

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int *vec_1 = (int *)malloc(BUFF_SIZE * sizeof(int));
	int *vec_2 = (int *)malloc(BUFF_SIZE * sizeof(int));

	if (my_rank == MASTER_RANK)
	{
		srand(time(0) + my_rank);

		for (int i = 0; i < BUFF_SIZE; i++)
		{
			vec_1[i] = rand() % MAX_VEC_VALUE;
			vec_2[i] = rand() % MAX_VEC_VALUE;
		}
	}

	int portion_size = BUFF_SIZE / comm_size;
	int *local_vec_1 = (int *)malloc(portion_size * sizeof(int));
	int *local_vec_2 = (int *)malloc(portion_size * sizeof(int));

	MPI_Scatter(vec_1, portion_size, MPI_INTEGER, local_vec_1, portion_size, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	MPI_Scatter(vec_2, portion_size, MPI_INTEGER, local_vec_2, portion_size, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(CONST_DELAY * my_rank);
	int local_result = 0;
	for (int i = 0; i < portion_size; i++)
	{
		printf("{%d} calc: %d*%d\n", my_rank, local_vec_1[i], local_vec_2[i]);
		local_result += local_vec_1[i] * local_vec_2[i];
	}

	printf("{%d} ----= %d\n", my_rank, local_result);
	printf("\n");

	int total_sum = 0;
	MPI_Reduce(&local_result, &total_sum, 1, MPI_INTEGER, MPI_SUM,
			   MASTER_RANK, MPI_COMM_WORLD);

	if (my_rank == MASTER_RANK)
	{
		usleep(CONST_DELAY * comm_size + 1);
		printf("{%d}vec_1->\t");
		for (int i = 0; i < BUFF_SIZE; i++)
		{
			printf("%d\t", vec_1[i]);
		}
		printf("\n");
		printf("{%d}vec_2->\t", my_rank);
		for (int i = 0; i < BUFF_SIZE; i++)
		{
			printf("%d\t", vec_2[i]);
		}
		printf("\n");

		printf("{%d} Total result is: %d\n", my_rank, total_sum);
	}

	free(local_vec_1);
	free(local_vec_2);

	free(vec_1);
	free(vec_2);

	MPI_Finalize();

	return 0;
}
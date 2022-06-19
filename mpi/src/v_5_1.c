#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define MAX_INT 99999

// Napisati MPI program koji pronalazi i prikazuje minimalni neparan broj sa zadatom
// osobinom i identifikator procesa koji ga sadrži.Neparni brojevi se nalaze u intervalu
// [a, b](a i b su zadate konstante). Osobina koju broj treba da poseduje je da je deljiv
// zadatom vrednošću x.Prilikom ispitivanja(da li broj poseduje zadatu osobinu ili ne)
// svaki proces generiše i ispituje odgovarajuće neparne brojeve na način prikazan na slici
// (za primer broj_procesa = 4 i a = 3, b = 31, x = 5). Konačne rezultate treba da prikaže
// proces koji sadrži najmanji broj takvih brojeva.Zadatak rešiti korišćenjem grupnih
// operacija.

// pronaci minimalni neparan broj deljiv brojem x
// konacna rezultat prikazati u procesu koji sadrzi najmanji broj takvih brojeva

/*
	p = 4
	a = 3
	b = 31
	x = 5

	P0	P1	P2	P3
	--------------
	3	5	7	9
	11	13	15	17
	19	21	23	25
	27	29	31

	p0 = |a _ _ _ _ _ _ _||* _ _ _ _ _ _ _||* _ _ _ _ _ _ _||* _ _ _ _ _ _ _|
	p1 = |_ * _ _ _ _ _ _||_ * _ _ _ _ _ _||_ * _ _ _ _ _ _||_ * _ _ _ _ _ _|
	p2 = |_ _ * _ _ _ _ _||_ _ * _ _ _ _ _||_ _ * _ _ _ _ _||_ _ * _ _ _ _ _|
	p3 = |_ _ _ * _ _ _ _||_ _ _ * _ _ _ _||_ _ _ b _ _ _ _||_ _ _ _ _ _ _ _|

*/

void init_vec(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

void generate_vector(int my_rank, int p_count, int *vec, int min, int max)
{
	int i = 0;
	while (vec[i - 1] < max)
	{
		int value = 2 * (i * p_count) + min + (2 * my_rank);
		if (value <= max)
		{
			vec[i] = value;
		}
		else
		{
			vec[i] = MAX_INT;
		}

		i++;
	}

	return;
}

void print_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		printf("%d \t", vec[i]);
	}
}

struct indexed_value
{
	int value;
	int index;
} typedef INDEXED_VALUE;

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

	int min_v_value = 3;
	int max_v_value = 100;
	int d = 5;

	int vec_len = ((max_v_value - min_v_value) / 2) / comm_size + 1;

	int *vec = (int *)malloc(vec_len * sizeof(int));
	generate_vector(my_rank, comm_size, vec, min_v_value, max_v_value);

	usleep(SLEEP_PERIOD * my_rank);
	printf("{%d} vector: ", my_rank);
	print_vec(vec, vec_len);
	printf("\n");

	int my_min = MAX_INT;
	int div_count = 0;
	for (int i = 0; i < vec_len; i++)
	{
		if (vec[i] != MAX_INT && vec[i] % d == 0)
		{
			div_count++;
			if (vec[i] < my_min)
			{
				my_min = vec[i];
			}
		}
	}

	INDEXED_VALUE indexed_min;
	indexed_min.index = my_rank;
	indexed_min.value = my_min;

	INDEXED_VALUE indexed_count;
	indexed_count.index = my_rank;
	indexed_count.value = div_count;

	INDEXED_VALUE real_min;

	MPI_Reduce(&indexed_min, &real_min, 1, MPI_2INTEGER, MPI_MINLOC, MASTER_RANK, MPI_COMM_WORLD);
	MPI_Bcast(&real_min, 1, MPI_2INTEGER, MASTER_RANK, MPI_COMM_WORLD);

	INDEXED_VALUE max_count;
	MPI_Reduce(&indexed_count, &max_count, 1, MPI_2INTEGER, MPI_MAXLOC, real_min.index, MPI_COMM_WORLD);

	if (my_rank == real_min.index)
	{
		printf("{%d} min value: %d from proc: %d\n", my_rank, real_min.value, real_min.index);
		printf("{%d} max count: %d from proc: %d\n", my_rank, max_count.value, max_count.index);
	}

	MPI_Finalize();

	return 0;
}

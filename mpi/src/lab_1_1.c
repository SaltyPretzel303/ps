#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Napisati MPI program koji pronalazi minimalnu vrednost u delu matrice reda n(n - parno)
// koga čine kolone matrice sa parnim indeksom(j = 0, 2, 4, ...).
// Matrica je inicijalizovana u master procesu(P0).
// Svaki proces treba da dobije elemente kolona sa parnim indeksom iz
// 	odgovarajućih n	/ p vrsta(p - broj procesa, n deljivo sa p) i nađe lokalni minimum.
// Na taj način,
// P0 dobija elemente kolona sa parnim indeksom iz prvih n / p vrsta i nalazi lokalni
// minimum,
// P1 dobija elemente kolona sa parnim indeksom iz sledećih n / p vrsta i nalazi
// lokalni minimum itd.Nakon toga,
// u master procesu se izračunava i na ekranu prikazuje
// globalni minimum u traženom delu matrice.Zadatak realizovati korišćenjem isključivo
// grupnih operacija i izvedenih tipova podataka.

#define TRUE 1
#define FALSE 0

#define MASTER_RANK 0
#define MPI_NO_ERROR 0
#define SLEEP_PERIOD 3000

void init_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = i;
	}
}

int *get_int_vec(int len)
{
	return (int *)malloc(len * sizeof(int));
}

void clear_vec(int *vec, int len)
{
	for (int i = 0; i < len; i++)
	{
		vec[i] = 0;
	}
}

void print_3_num(int num)
{
	if (num < 10)
	{
		printf("0");
	}
	if (num < 100)
	{
		printf("0");
	}
	printf("%d", num);
}

void print_vec_as_mat(int *vec, int r_dim, int c_dim, int print_zeros)
{
	for (int i = 0; i < r_dim * c_dim; i++)
	{
		if (i % r_dim == 0)
		{
			printf("\n");
		}
		if (vec[i] == 0)
		{
			if (!print_zeros)
			{
				printf("---");
			}
			else
			{
				print_3_num(vec[i]);
			}
		}
		else
		{
			print_3_num(vec[i]);
		}
		printf(" | ");
	}
	printf("\n");
}

void print_vec(int *vec, int len, int print_zeros)
{
	for (int i = 0; i < len; i++)
	{
		if (vec[i] == 0)
		{
			if (!print_zeros)
			{
				printf("---");
			}
			else
			{
				print_3_num(vec[i]);
			}
		}
		else
		{
			print_3_num(vec[i]);
		}
		printf(" | ");
	}
	printf("\n");
}

int is_master(int my_rank)
{
	return my_rank == MASTER_RANK;
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	int mat_dim = 16;

	int *mat;
	if (is_master(my_rank))
	{
		mat = get_int_vec(mat_dim * mat_dim);
		init_vec(mat, mat_dim * mat_dim);
		print_vec_as_mat(mat, mat_dim, mat_dim, TRUE);
	}

	int bl_cnt = mat_dim;
	int bl_size = 1;
	int bl_stride = mat_dim;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype col_type;
	MPI_Type_vector(bl_cnt, bl_size, bl_stride, old_type, &col_type);
	MPI_Type_commit(&col_type);

	int bl = 0;
	int extent = 2 * sizeof(int);
	MPI_Datatype rs_col_type;
	MPI_Type_create_resized(col_type, bl, extent, &rs_col_type);
	MPI_Type_commit(&rs_col_type);

	int rec_cnt = mat_dim * (mat_dim / 2) / comm_size;
	int *rec_cols = get_int_vec(rec_cnt);

	MPI_Scatter(mat, rec_cnt / mat_dim, rs_col_type,
				rec_cols, rec_cnt, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	usleep(my_rank * SLEEP_PERIOD * 2);
	printf("{%d} (%d) -> %d vec: ", my_rank, rec_cnt, rec_cnt / mat_dim);
	print_vec(rec_cols, rec_cnt, TRUE);

	int loc_min = 99999; // fake max value
	for (int i = 0; i < rec_cnt; i++)
	{
		if (rec_cols[i] < loc_min)
		{
			loc_min = rec_cols[i];
		}
	}

	int all_min = 9999;
	MPI_Reduce(&loc_min, &all_min, 1, MPI_INTEGER, MPI_MIN, MASTER_RANK, MPI_COMM_WORLD);

	if (is_master(my_rank))
	{
		usleep((comm_size + 5) * SLEEP_PERIOD);
		printf("{%d} allMin: %d \n\n", my_rank, all_min);
	}

	MPI_Finalize();

	return 0;
}
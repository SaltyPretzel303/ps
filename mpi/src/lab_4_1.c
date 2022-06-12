#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

// Napisati MPI program koji realizuje množenje matrice A mxn i matrice B nxk, čime se
// dobija rezultujuća matrica C mxk.Množenje se obavlja tako što master proces šalje svakom
// procesu celu matricu A i po k /
// p kolona matrice B(p - broj procesa, k je deljivo sa p).Svi procesi učestvuju u izračunavanju.
// Konačni rezultat množenja se nalazi u master procesu koji ga i prikazuje.
// Predvideti da se slanje k / p kolona matrice B svakom procesu obavlja odjednom i
// to direktno iz matrice B.Zadatak rešiti korišćenjem grupnih operacija i
// izvedenih tipova podataka.

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
		if (i % c_dim == 0)
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

	int mat_dim = 4;

	int *mat_a;
	mat_a = get_int_vec(mat_dim * mat_dim);
	int *mat_b;
	int *mat_c;
	if (is_master(my_rank))
	{

		init_vec(mat_a, mat_dim * mat_dim);
		print_vec_as_mat(mat_a, mat_dim, mat_dim, TRUE);
		printf("\n");

		mat_b = get_int_vec(mat_dim * mat_dim);
		init_vec(mat_b, mat_dim * mat_dim);
		print_vec_as_mat(mat_b, mat_dim, mat_dim, TRUE);

		mat_c = get_int_vec(mat_dim * mat_dim);
		printf("=============================================\n");
	}

	int s_bl_cnt = mat_dim;
	int s_bl_size = 1;
	int s_bl_stride = mat_dim;
	MPI_Datatype old_type = MPI_INTEGER;
	MPI_Datatype col_type;
	MPI_Type_vector(s_bl_cnt, s_bl_size, s_bl_stride, old_type, &col_type);
	MPI_Type_commit(&col_type);

	int bl = 0;
	int extent = 1 * sizeof(int);
	MPI_Datatype rs_col_type;
	MPI_Type_create_resized(col_type, bl, extent, &rs_col_type);
	MPI_Type_commit(&rs_col_type);

	int col_cnt = mat_dim / comm_size;
	int *r_vec = get_int_vec(col_cnt * mat_dim);

	MPI_Scatter(mat_b, col_cnt, rs_col_type,
				r_vec, col_cnt * mat_dim, MPI_INTEGER,
				MASTER_RANK, MPI_COMM_WORLD);

	MPI_Bcast(mat_a, mat_dim * mat_dim, MPI_INTEGER, MASTER_RANK, MPI_COMM_WORLD);

	int loc_res_cnt = col_cnt * mat_dim;
	int *local_res = get_int_vec(loc_res_cnt);
	clear_vec(local_res, loc_res_cnt);

	for (int i = 0; i < mat_dim; i++)
	{
		for (int j = 0; j < col_cnt; j++)
		{
			for (int k = 0; k < mat_dim; k++)
			{
				int res_ind = j * mat_dim + i;
				int a_ind = i * mat_dim + k;
				int b_ind = j * mat_dim + k;

				local_res[res_ind] += mat_a[a_ind] * r_vec[b_ind];
			}
		}
	}

	MPI_Gather(local_res, col_cnt * mat_dim, MPI_INTEGER,
			   mat_c, col_cnt, rs_col_type, MASTER_RANK, MPI_COMM_WORLD);

	usleep(my_rank * 2 * SLEEP_PERIOD);
	printf("{%d} r_vec: ", my_rank);
	print_vec(r_vec, col_cnt * mat_dim, TRUE);
	print_vec_as_mat(local_res, col_cnt, mat_dim, TRUE);

	if (is_master(my_rank))
	{
		usleep(comm_size * 3 * SLEEP_PERIOD);

		print_vec_as_mat(mat_c, mat_dim, mat_dim, TRUE);
	}

	MPI_Finalize();

	return 0;
}
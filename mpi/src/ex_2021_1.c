#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>
#include <math.h>

// Pomnoziti matricu A(KxK) i vektor B(K) cime se dobija vektor C(K).
// Matrica A i vektor B se inicijalizuju u master procesu.
// Broj procesa je P i oni su uredjeni u vidu matrice prikazane na slici pri cemu je
// velicina matrice QxQ i vazi da je Q^2=p => Q=sqrt(P) (na slici je matica za K=8 i P=16)
// Matrica A je podeljena u blokove i master proces distribuira odgovarajuce blokove matrice
// kao sto je prikazano na slici. Vektor B je distribuiran tako da proces Pi dobija elemente
// sa indeksima (i%q)*s, (i%q)*s+1, (i%q)*s+2 ... (i%q)*s+s-1 s=k/1.
// Predvideti da se slanje vrednosti bloka mtarice A svakom procesu obavlja odjednom.
// Svaki proces obavlja izracunavanja i ucestvuje u generisanju rezultata koji se prikazuje
// u procesu koji sadrzi minimum svih vrednosti u matrici A. Slanje blokova A i vektora
// B kaoi generisanje rezultata implementirati koriscenjem grupnih operacija i funkcija
// za kreiranje novih komunikator

// "SLIKA":
/*

	a00		a01		a02		a03		a04		a05		a06		a07
		P0				P1				P2				P3
	a40		a41		a42		a43		a44		a45		a46		a47


		*				*				*				*



		*				*				*				*


	a30		a31		a32		a33		a34		a35		a36		a37
		P12				P13				P14				P15
	a70		a71		a72		a73		a74		a75		a76		a77  (Q*Q) <- dimension


*/

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

	int MAT_DIM = 8;
	int VEC_DIM = 8;

	int q = sqrt(comm_size);
	int s = MAT_DIM / q;

	int *mat = get_int_vec(MAT_DIM * MAT_DIM);
	int *vec = get_int_vec(VEC_DIM);

	int *loc_mat = get_int_vec(MAT_DIM * MAT_DIM);
	int *loc_vec = get_int_vec(MAT_DIM);
	clear_vec(loc_mat, MAT_DIM * MAT_DIM);
	clear_vec(loc_vec, MAT_DIM);

	if (is_master(my_rank))
	{
		init_vec(mat, MAT_DIM * MAT_DIM);
		init_vec(vec, VEC_DIM);

		print_vec_as_mat(mat, MAT_DIM, MAT_DIM, FALSE);
		printf("\n");
		print_vec(vec, VEC_DIM, FALSE);
		printf("\n");
	}

#pragma region scatter matrix blocks

	int mbl_bl_cnt = s;
	int mbl_bl_size = s;
	int mbl_bl_stride = s * (q / 2) * MAT_DIM;
	MPI_Datatype mbl_old_type = MPI_INTEGER;
	MPI_Datatype mat_block;
	MPI_Datatype rs_mat_block;

	MPI_Type_vector(mbl_bl_cnt, mbl_bl_size, mbl_bl_stride, mbl_old_type, &mat_block);
	MPI_Type_commit(&mat_block);
	int bl = 0;
	int extent = s * sizeof(int);
	MPI_Type_create_resized(mat_block, bl, extent, &rs_mat_block);
	MPI_Type_commit(&rs_mat_block);

	MPI_Scatter(mat, 1, rs_mat_block,
				&loc_mat[s * my_rank], 1, rs_mat_block,
				MASTER_RANK, MPI_COMM_WORLD);

#pragma endregion

#pragma region scatter vector blocks

	// create group/comm containing elements of the first row
	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);
	MPI_Group masters_group;
	MPI_Comm masters_comm;

	int master_cnt = s;
	int *master_indices = get_int_vec(master_cnt);
	for (int i = 0; i < q; i++)
	{
		master_indices[i] = i;
	}
	MPI_Group_incl(world_group, master_cnt, master_indices, &masters_group);
	MPI_Comm_create(MPI_COMM_WORLD, world_group, &masters_comm);

	// actually i don't need custom type for this type of scatter
	// basically every process is gonna get sequence of S elements
	// int vbl_bl_cnt = 1;
	// int vbl_bl_size = s;
	// int vbl_bl_stride = s;
	// MPI_Datatype vbl_old_type = MPI_INTEGER;
	// MPI_Datatype vec_block;
	// MPI_Datatype rs_vec_block;

	// MPI_Type_vector(vbl_bl_cnt, vbl_bl_size, vbl_bl_stride, vbl_old_type, &vec_block);
	// MPI_Type_commit(&vec_block);

	// MPI_Scatter(vec, 1, vec_block,
	// 			&loc_vec[s * my_rank], 1, vec_block,
	// 			MASTER_RANK, MPI_COMM_WORLD);

	MPI_Scatter(vec, s, MPI_INTEGER,
				&loc_vec[s * my_rank], s, MPI_INTEGER,
				MASTER_RANK, masters_comm);

	// create separate group/comm for each column so that master of each column
	// (first element) can broadcast vector block to other processes
	// if order is kept from the WORLD_COMM processes from the first row
	// should have rank equals to MASTER_RANK
	int color = my_rank % q;
	MPI_Comm col_com;
	MPI_Comm_split(MPI_COMM_WORLD, color, 0, &col_com);

	// sending the whole vector is unnecessary but ... whatever
	MPI_Bcast(loc_vec, VEC_DIM, MPI_INTEGER, MASTER_RANK, col_com);

#pragma endregion

#pragma region calculation and results gather

	int *loc_res = get_int_vec(MAT_DIM * MAT_DIM);
	clear_vec(loc_res, MAT_DIM * MAT_DIM);

	int row = my_rank / q;
	int row_step = q;

	int col_step = 1;

	for (int i = 0; i < s; i++)
	{
		int col = s * (my_rank % q);
		for (int j = 0; j < s; j++)
		{
			loc_res[MAT_DIM * row + col] = loc_mat[MAT_DIM * row + col] * loc_vec[col];
			col += col_step;
		}
		row += row_step;
	}

	int *final_vec = get_int_vec(VEC_DIM);

	int rdc_bl_cnt = s;
	int rdc_bl_size = MAT_DIM;
	int rdc_bl_stride = -1;
	MPI_Datatype reduce_vec_el = MPI_INTEGER;
	MPI_Datatype reduce_vec;

	MPI_Reduce(loc_mat, final_vec, VEC_DIM, MPI_INTEGER, MPI_SUM, MASTER_RANK, MPI_COMM_WORLD);

	if (is_master(my_rank))
	{
		usleep(SLEEP_PERIOD * comm_size * 10);
		printf("result: ");
		print_vec(final_vec, VEC_DIM, FALSE);
		printf("\n");
	}

#pragma endregion

	usleep(SLEEP_PERIOD * 2 * my_rank);
	print_vec_as_mat(loc_mat, MAT_DIM, MAT_DIM, FALSE);
	printf("\n");
	print_vec_as_mat(loc_res, MAT_DIM, MAT_DIM, FALSE);
	printf("\n");
	print_vec(loc_vec, VEC_DIM, FALSE);
	printf("\n");

	printf("===================================");
	printf("\n");

#pragma region free memory

	free(mat);
	free(vec);
	free(loc_mat);
	free(loc_vec);
	free(master_indices);

#pragma endregion

	MPI_Finalize();

	return 0;
}

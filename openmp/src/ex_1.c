#include <stdio.h>
#include <omp.h>

#define VEC_DIM 10

void print_vec(int *vec, int dim)
{
	for (int i = 0; i < dim; i++)
	{
		printf("%d, ", vec[i]);
	}

	printf("\n");
}

int main(void)
{

	int vec_a[VEC_DIM];
	int vec_b[VEC_DIM];
	int vec_c[VEC_DIM];
	int vec_d[VEC_DIM];

	int res_vec[VEC_DIM];

	for (int i = 0; i < VEC_DIM; i++)
	{
		vec_a[i] = i;
		vec_b[i] = 1;
		vec_c[i] = 1;
		vec_d[i] = 0;

		res_vec[i] = 0;
	}

#pragma omp parallel for firstprivate(vec_a)

	for (int i = 0; i < VEC_DIM - 1; i++)
	{
		int x = vec_b[i] + vec_c[i];

		res_vec[i] = vec_a[i + 1] + x;
	}

	print_vec(vec_a, VEC_DIM);
	print_vec(res_vec, VEC_DIM);

	printf("==========================\n");

	int x = 0;

	for (int i = 0; i < VEC_DIM; i++)
	{
		x = (vec_b[i] + vec_c[i]) / 2;
		vec_a[i] = vec_a[i] + x;

		vec_d[1] = 2 * x;
	}

	int y = x + vec_d[1] + vec_d[2];

	printf("==========================\n");

	int sum = 0;
	for (int i = 0; i < VEC_DIM; i++)
	{
		sum = sum + vec_a[i];
	}

	printf("control sum: %d\n", sum);

	int red_sum = 0;

#pragma omp parallel for reduction(+ \
								   : red_sum)
	for (int i = 0; i < VEC_DIM; i++)
	{
		red_sum = vec_a[i];
	}

	printf("red_sum: %d ... \n", red_sum);

	printf("==========================\n");

	for (int i = 0; i < VEC_DIM; i++)
	{
		vec_b[i] += vec_a[i - 1];
		vec_a[i] += vec_c[i];
	}

	return 0;
}

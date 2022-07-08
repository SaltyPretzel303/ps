#include <stdio.h>
#include <omp.h>

int main(void)
{

	int dim = 10;

#pragma omp parallel for num_threads(dim)

	for (int i = 0; i < dim; i++)
	{
		int thread_id = omp_get_thread_num();

		printf("i=%d, id=%d \n", i, thread_id);
	}

	return 0;
}

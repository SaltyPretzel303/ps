#include <stdio.h>
#include <omp.h>

int main()
{

	int a = 0;

	omp_set_num_threads(4);
#pragma omp parallel for private(a) ordered schedule(guided, 1)
	for (int i = 0; i < 40; i++)
	{
		int id = omp_get_thread_num();
#pragma omp ordered
		printf("My id: \t%d i: \t%d \n", id, i);
	}

	printf("In master -> :a = %d \n", a);

	return 1;
}
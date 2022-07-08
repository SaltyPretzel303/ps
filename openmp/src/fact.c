#include <stdio.h>

int main()
{

	int fact = 1;

	int end = 4;

	// for (int i = 2; i < end; i++)
	// {
	// 	fact += i;
	// }

	// 1 2 3 4 5 

#pragma omp parallel for reduction(* \
								   : fact)
	for (int i = 1; i <= end; i++)
	{
		fact = i;
	}

	printf("fact: %d\n", fact);

	return 0;
}
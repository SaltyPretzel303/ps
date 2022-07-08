#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void kernel(cudaEvent_t s_event)
{
	return;
}

int main(void)
{

	cudaEvent_t s_event;

	if (cudaEventQuery(s_event) == cudaSuccess)
	{
		printf("Before record success ... \n");
	}
	else
	{
		printf("Before record failed ... \n");
	}

	// cudaEventRecord(s_event);

	kernel<<<1, 1>>>(s_event);

	cudaEventCreate(&s_event);

	cudaEventRecord()

		if (cudaEventQuery(s_event) == cudaSuccess)
	{
		printf("After record success ... \n");
	}
	else
	{
		printf("After record failed ... \n");
	}

	return 0;
}

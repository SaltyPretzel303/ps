#include <stdio.h>

__global__ void kernel_add(int a, int b, int *c)
{
	*c = a + b;
}

int main(void)
{

	int value_a = 2;
	int value_b = 7;

	int value_c = -1;
	int *dev_c;

	cudaMalloc((void **)&dev_c, sizeof(int));

	kernel_add<<<1, 1>>>(2, 7, dev_c);

	cudaMemcpy(&value_c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Hello cuda world: %d + %d = %d ... \n",
		   value_a, value_b, value_c);

	cudaFree(dev_c);

	return 0;
}
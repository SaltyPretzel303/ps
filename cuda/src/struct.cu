#include <stdio.h>
#include "cuda_help.h"

struct two_values
{
	int int_value;
	float float_value;
} typedef two_values;

__global__ void kernel(two_values *values)
{
	(*values).int_value += 2;
	(*values).float_value += 2.2;

	return;
}

int main(void)
{

	two_values values;
	values.int_value = 10;
	values.float_value = 20;

	two_values *dev_values;
	cuda_err(cudaMalloc((void **)&dev_values, sizeof(two_values)),
			 "devTwoValues cudaMalloc");

	cuda_err(cudaMemcpy(dev_values, &values, sizeof(two_values), cudaMemcpyHostToDevice),
			 "devTwoValues cudaMemcpy to device");

	kernel<<<1, 1>>>(dev_values);

	cuda_err(cudaMemcpy(&values, dev_values, sizeof(two_values), cudaMemcpyDeviceToHost),
			 "devTwoValues cudaMemcpy to host");

	printf("int_value: %d, float_value: %f ... \n", values.int_value, values.float_value);

	cudaFree(dev_values);

	cudaDeviceReset();

	return 0;
}
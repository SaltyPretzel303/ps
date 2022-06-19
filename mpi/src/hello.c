#include <stdio.h>
#include <stdlib.h>

int main()
{
	const int BUFF_SIZE = 20;

	int *send_buff;
	send_buff = malloc(BUFF_SIZE * sizeof(int));

	int i = 0;
	for (i = 0; i < BUFF_SIZE; i++)
	{
		send_buff[i] = i;
	}

	for (i = 0; i < BUFF_SIZE; i++)
	{
		printf("buff[%d] => %i\n", i, send_buff[i]);
	}

	free(send_buff);

	return 0;
}
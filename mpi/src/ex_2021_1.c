#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // usleep
#include <mpi.h>

#define TRUE 1
#define FALSE 0

// Pomnoziti matricu A i vektor B cime se dobija vektor C. 
// Matrica A i vektor B se inicijalizuju u mater procesu. 
// Broj procesa je P i oni su poredjani u vidu matrice prikazane na slici pri cemu je 
// velicina matrice QxQ i vazi da je q^2=p. Matrica A je podeljena u blokove i master 
// proces distribuira odgovarajuce blokove matrice tako da proces Pi dobija elemente 
// indeksima (i%q)*s, (i%q)*(s+1), (i%q)*(s+2) ... (i%q)*(s+s-1) s=k/1.
// Predvideti da se slanje vrednosti bloka mtarice A svakom procesu obavlja odjednom. 
// Svaki proces obavlja izracunavanja i ucestvuje u generisanju rezultata koji se prikazuje
// u procesu koji sadrzi minimum svih vrednosti u matrici A. Slanje blokova A i vektora
// B kaoi generisanje rezultata implementirati koriscenjem grupnih operacija i funkcija
// za kreiranje novih komunikator

// TODO translate

void init_buff(int *buff, int len)
{
	for (int i = 0; i < len; i++)
	{
		buff[i] = i;
	}
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	const int BUFF_SIZE = 20;
	const int MASTER_RANK = 0;
	const int MPI_NO_ERROR = MPI_SUCCESS;
	const int SLEEP_PERIOD = 3000;

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	

	MPI_Finalize();

	return 0;
}
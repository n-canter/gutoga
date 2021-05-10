all:
	gcc -c -Wall -Werror -fpic gutoga.c sha256.c -fopenmp -O3
	gcc -shared -o libgutoga.so sha256.o gutoga.o

clean:
	rm *.o libgutoga.so

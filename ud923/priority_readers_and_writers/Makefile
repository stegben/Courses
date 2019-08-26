CC=gcc

all: read_priority_int.o
	$(CC) -c main main.c read_priority_int.o

read_priority_int.o:
	$(CC) -c read_priority_int.h -o read_priority_int.o -pthread

clean:
	rm -rf *.o *.h.gch main

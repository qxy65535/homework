CC = gcc
CFLAGS = -O3 -Wall -g -msse
LDFLAGS = -lm 

all: c63enc c63dec c63pred

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) -DC63_PRED $(CFLAGS) $^ -o $@ $(LDFLAGS)
	
clean:
	rm -f *.o c63enc c63dec c63pred test

test: test.o
	$(CC) $^ -o $@
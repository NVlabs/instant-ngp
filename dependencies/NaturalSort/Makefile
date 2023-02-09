CC = g++
CFLAGS = -Wall

test: natural_sort_test.cpp natural_sort.hpp
	$(CC) $(CFLAGS) -o $@ $<
	./$@

clean:
	rm -f test

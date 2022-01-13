include ../Config

TARGETS = diffusion-raw diffusion-zfp pgm simple speed testzfp zfp

all: $(TARGETS)

diffusion-raw: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -DWITHOUT_COMPRESSION -I../array diffusion.cpp -L../lib -lzfp -o diffusion-raw

diffusion-zfp: diffusion.cpp ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array diffusion.cpp -L../lib -lzfp -o diffusion-zfp

pgm: pgm.c ../lib/libzfp.a
	$(CC) $(CFLAGS) pgm.c -L../lib -lzfp -lm -o pgm

simple: simple.c ../lib/libzfp.a
	$(CC) $(CFLAGS) simple.c -L../lib -lzfp -lm -o simple

speed: speed.c ../lib/libzfp.a
	$(CC) $(CFLAGS) speed.c -L../lib -lzfp -lm -o speed

testzfp: testzfp.cpp fields.o ../lib/libzfp.a
	$(CXX) $(CXXFLAGS) -I../array testzfp.cpp fields.o -L../lib -lzfp -o testzfp

fields.o: fields.c
	$(CC) $(CFLAGS) -c fields.c

zfp: zfp.c ../lib/libzfp.a
	$(CC) $(CFLAGS) zfp.c -L../lib -lzfp -lm -o zfp

test: testzfp
	./testzfp

clean:
	rm -f $(TARGETS) fields.o

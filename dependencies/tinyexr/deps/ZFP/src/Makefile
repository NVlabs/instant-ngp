include ../Config

TARGETS = ../lib/libzfp.a ../lib/libzfp.so
OBJECTS = bitstream.o decode1f.o decode1d.o encode1f.o encode1d.o decode2f.o decode2d.o encode2f.o encode2d.o decode3f.o decode3d.o encode3f.o encode3d.o zfp.o

../lib/libzfp.a: $(OBJECTS)
	rm -f $@
	ar rc $@ $^

../lib/libzfp.so: $(OBJECTS)
	$(CC) $(CFLAGS) -shared $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $<

static: ../lib/libzfp.a

shared: ../lib/libzfp.so

clean:
	rm -f $(TARGETS) $(OBJECTS)

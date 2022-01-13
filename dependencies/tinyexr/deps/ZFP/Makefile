MAKEFLAGS += --no-print-directory

all:
	@cd src; $(MAKE) clean static
	@cd examples; $(MAKE) clean all

shared:
	@cd src; $(MAKE) shared

test:
	@cd examples; $(MAKE) test

clean:
	@cd src; $(MAKE) clean
	@cd examples; $(MAKE) clean


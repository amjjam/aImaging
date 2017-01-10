libdir=$(prefix)/lib

CPPFLAGS=-Wall -g -fPIC -I ~/packages/atime/include -I ~/packages/aVec/include -I /usr/lib/openmpi/include

CPP=g++

AIMAGING_SRC=image.C rimage.C camera.C rcamera.C losGslQag.C
AIMAGING_OBJ=$(AIMAGING_SRC:.C=.o)

build: libaImaging.a libaImaging.so

libaImaging.a: $(AIMAGING_OBJ)
	ar rc $@ $^

libaImaging.so: $(AIMAGING_OBJ)
	$(CPP) -shared -Wl,-soname,$@ -o $@ $^

install: $(libdir)/libaImaging.a $(libdir)/libaImaging.so

$(libdir)/libaImaging.a $(libdir)/libaImaging.so: libaImaging.a libaImaging.so
	cp libaImaging.a libaImaging.so $(libdir)

uninstall:
	- rm -f $(libdir)/libaImaging.a $(libdir)/libaImaging.so

clean:
	- rm -f libaImaging.a libaImaging.so $(AIMAGING_OBJ)

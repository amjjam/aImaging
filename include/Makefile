includedir=$(prefix)/include

INCLUDE_FILES=image.H rimage.H camera.H rcamera.H model.H los.H losGslQag.H

build: aImaging.H

aImaging.H: $(INCLUDE_FILES)
	cat $^ > $@

install: $(includedir)/aImaging.H

$(includedir)/aImaging.H: aImaging.H
	cp $^ $@

uninstall:
	- rm -f $(includedir)/aImaging.H

clean:
	- rm -f aImaging.H

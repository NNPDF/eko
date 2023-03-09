all: matching_and_rotation.pdf

%.pdf: %.tex
	pdflatex $<

clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.pdf

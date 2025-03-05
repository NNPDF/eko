# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

EKODIR = ../src/eko
EKOOUT = $(SOURCEDIR)/modules/eko
EKOMARKDIR = ../src/ekomark
EKOMARKOUT = $(SOURCEDIR)/development/ekomark
EKOBOXKDIR = ../src/ekobox
EKOBOXKOUT = $(SOURCEDIR)/code/ekobox
EKOREOUT = $(SOURCEDIR)/modules/ekore

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html view clean cleanall

# sphinx
html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

server:
	python3 -m http.server 8000 --bind 127.0.0.1 --directory build/html

view: html
	$(BROWSER) $(BUILDDIR)/html/index.html

clean:
	rm -rf build
	rm -rf _build

cleanall: clean
	rm -rf $(EKOOUT)
	rm -rf $(EKOMARKOUT)
	rm -rf $(EKOBOXKOUT)
	rm -rf $(EKOREOUT)

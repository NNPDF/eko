# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

EKODIR = ../src/eko
EKOTESTDIR = ../src/eko/tests

TODOOUTFILE = ./source/code_todos.rst

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html view clean cleanall todos clean-todos

# sphinx
html: todos
	sphinx-apidoc --module-first -o ./source/modules/eko "$(EKODIR)" "$(EKOTESTDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

view: html
	$(BROWSER) $(BUILDDIR)/html/index.html

clean:
	rm -rf build
	rm -rf _build

cleanall: clean clean-todos
	rm -rf $(SOURCEDIR)/modules

# TODOs
todos:
	python generate_code_todos.py "$(EKODIR)" "$(TODOOUTFILE)"

clean-todos:
	rm "$(TODOOUTFILE)"

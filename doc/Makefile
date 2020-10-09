# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

EKODIR = ../src/eko
EKOOUT = $(SOURCEDIR)/modules/eko
EKOMARKDIR = ../benchmarks/ekomark
EKOMARKOUT = ./source/development/ekomark

TODOOUTFILE = ./source/development/code_todos.rst

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html view clean cleanall todos clean-todos

# sphinx
html: todos
	sphinx-apidoc --module-first -o "$(EKOOUT)" "$(EKODIR)"
	@rm -f $(EKOOUT)/modules.rst
	sphinx-apidoc --module-first -o "$(EKOMARKOUT)" "$(EKOMARKDIR)"
	@rm -f $(EKOMARKOUT)/modules.rst
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

# -*- coding: utf-8 -*-

import argparse
import pathlib
import re

def get_all_files(path):
    """
        Find all .py files in `path`

        Parameters
        ----------
            path : pathlib.Path
                source directory

        Returns
        -------
            files : list
                list of files
    """
    files = path.glob("*.py")
    return list(files)

def search_file(path):
    """
        Search all todo tags in the file

        Parameters
        ----------
            path : pathlike
                path to input file

        Returns
        -------
            todos : list
                list of tuples with (linenumber, text)
    """
    todos = []
    with open(path,"r") as o:
        j = 1
        for l in o:
            m = re.search(r'\s*#\s+TODO\s+(.+)$',l) # TODO grep sourdings?
                                                    # i.e. if multiple lines?
            if m is not None:
                todos.append((j,m.group(1)))
            j += 1
    return todos

def write_output(fn, file_list):
    """
        Writes the list to a file

        Parameters
        ----------
            fn : string
                output file name
            file_list : list
                list of todos sorted by file
    """
    with open(fn,"w") as o:
        # head
        o.write("TODOs\n")
        o.write("="*5+"\n\n")
        o.write("Fom Docstrings\n")
        o.write("-"*14+"\n\n")
        o.write(".. todolist ::\n\n")
        o.write("Fom Source Code\n")
        o.write("-"*15+"\n\n")
        for path,todos in file_list:
            # file head
            # TODO use something better then file://? but this should never appear in production
            o.write("`"+str(path.name)+" <file://"+str(path.resolve())+">`_\n\n")
            for ln,todo in todos:
                o.write(f".. warning:: #{ln} {todo}\n\n")
            o.write("\n\n")

if __name__ == "__main__":
    # Mini argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_dir",
        default=None,
        type=str,
        help="scanned input directory.",
    )
    parser.add_argument(
        "output",
        default=None,
        type=str,
        help="path to output file.",
    )
    args = parser.parse_args()
    # find files
    path = pathlib.Path(args.src_dir)
    files = get_all_files(path)
    # add myself
    files.append(pathlib.Path(__file__))
    # find TODOS in input
    file_list = []
    for f in files:
        todos = search_file(f)
        # found somehting?
        if len(todos) < 1:
            continue
        file_list.append((f,todos))
    # write to file
    write_output(args.output,file_list)

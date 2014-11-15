Place uncompiled graphviz dot files in this directory.  Generate the dot files
by running `python profile.py ./test.py`

Probably should not commit PDF files to git

Give .dot files names that are indicative of the config under which they were
run. Also, leave comments "//" at the top of the file to list the exact relevant
configuration that the data was run under.

For each .dot file there is also a .pstats file.  Make sure that the two files
have the SAME NAME and just different extensions (.dot and .pstats)

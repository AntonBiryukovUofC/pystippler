import os
import sys
import timeit

def test():
    cmd = sys.executable
    os.system(cmd + " " + "src/rougier/stippler.py /home/anton/Repos/pystippler/data/obama.png \
               --n_point 15000 --n_iter 10 --pointsize 1.0 1.0 --figsize 10 \
               --force")
    return 0
print(timeit.timeit("test()", setup="from __main__ import test",number=1))
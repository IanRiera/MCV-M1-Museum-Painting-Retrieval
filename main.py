import sys

from week1 import run_w1 as w1
from week2 import run_w2 as w2
from week3 import run_w3 as w3
from week4 import run_w4 as w4
from week5 import run_w5 as w5

if __name__=="__main__":

    if "--week1" in sys.argv:
        w1.run()

    if "--week2" in sys.argv:
        w2.run()

    if "--week3" in sys.argv:
        w3.run()

    if "--week4" in sys.argv:
        w4.run()

    if "--week5" in sys.argv:
        w5.run()

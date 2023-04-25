# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2023-02-09 17:23:15
# @Last Modified: 2023-02-10 09:50:14
# ------------------------------------------------------------------------------ #
# Thin wrapper to run the branching network from command line
# pass arguments to the constructor like this" `run_bn.py --N=100 --k=10`
# ------------------------------------------------------------------------------ #

import os
import sys
import argparse

os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.abspath("../"))

import bn_with_correlated_input as bnm

bnm.log.setLevel("DEBUG")
# bnm.disable_tqdm()

parser = argparse.ArgumentParser()
_, args = parser.parse_known_args()

for arg in args:
    print(arg)
# remove leading dashes
args = [arg.lstrip("--") for arg in args]
args = [arg.lstrip("-") for arg in args]

# create a dict we can pass to the constructor
kwargs = {k: v for k, v in [arg.split("=") for arg in args]}

# try to cast to float, or to boolean if = "True" or "False"
for k, v in kwargs.items():
    if v == "True" or v == "False":
        kwargs[k] = eval(v)
    else:
        try:
            kwargs[k] = float(v)
        except ValueError:
            pass

# we are not super careful with casting here. passing debug=False will not work.
# use debug=0, and double check for other issues
bn = bnm.BranchingNetwork(**kwargs)
bn.run()




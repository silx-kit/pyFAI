#!/usr/bin/env python3

"""Tool to re-run all notebooks in the git repo in prevision of a release
"""
__author__ = "Jerome Kieffer"
__date__ = "14/06/2026"

import time
import os
import json
import sys
import subprocess
from datetime import timedelta


def get_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_ipynb(root):
    notebooks = []
    proc = subprocess.run(["git", "ls-files"], capture_output=True, cwd=root)
    assert proc.returncode == 0
    for f in proc.stdout.decode().split():
        if f.endswith(".ipynb") and "checkpoint" not in f and "-Copy" not in f:
            notebooks.append(os.path.join(root, f))
    return notebooks


def check_interactive(notebook):
    inter = False
    with open(notebook) as fd:
        nb = json.load(fd)
    for k in nb["cells"]:
        if inter:
            break
        for src in k["source"]:
            if src.startswith("%matplotlib widget"):
                inter = True
                break
    return inter


def run_notebook(fn):
    print(f"|{fn:100s} | ", end="")
    sys.stdout.flush()
    t1 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                           fn], capture_output=True)
    if proc.returncode != 0:
        print(f"{str(timedelta(seconds=time.perf_counter()-t1))[:-3]:20s} | {'FAILED':10s} |")
        with open(fn+".stdout", "wb") as stdout:
            stdout.write(proc.stdout)
        with open(fn+".stderr", "wb") as stderr:
            stderr.write(proc.stderr)
        return True
    else:
        print(f"{str(timedelta(seconds=time.perf_counter()-t1))[:-3]:20s} | {'':10s} |")


if __name__ == "__main__":
    t0 = time.perf_counter()
    root = get_root()
    print("Working directory:", root)
    print("*"*50)
    interactive = {n:check_interactive(n) for n in get_ipynb(root)}
    print("Interactive:\n"+"\n".join(i for i in interactive if interactive[i] is True))
    print("_"*139)
    print(f"|{'Path to non interactive notebook':100s} | {'Timing':20s} | {'Error ?':10s} |")
    print("_"*139)
    failed = [ fn for fn in interactive if not interactive[fn] and run_notebook(fn)]
    print("_"*139)
    print("Failed:\n"+"\n".join(failed))
    print("*"*50)
    print(f"Runtime: {timedelta(seconds=round(time.perf_counter()-t0))}s")

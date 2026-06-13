#!/usr/bin/env python3

import time
import os
import json
import sys
import subprocess

t0 = time.perf_counter()
notebooks = []
interactive = {}
failed = []

usage = os.path.dirname(os.path.abspath(__file__))
print(usage)
for root, dirs, files in os.walk(usage):
    for f in files:
        if f.endswith(".ipynb") and not f.startswith(".") and not "checkpoint" in f and not "-Copy" in f:
           notebooks.append(os.path.join(usage,root,f))

for f in notebooks:
    inter = False
    with open(f) as fd:
        nb = json.load(fd)
    for k in nb["cells"]:
        for src in k["source"]:
           if src.startswith("%matplotlib widget"):
               inter = True
    interactive[f] = inter

for k,v in interactive.items():
    if v is False:
        print(f"Rerun notebook {k} ", end="")
        sys.stdout.flush()
        t1 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                               k], capture_output=True
                              )
        if proc.returncode != 0:
            failed.append(k)
            print(f"t={time.perf_counter()-t1:.3f}s FAILED")
            print(proc.stdout)
            #rint(proc.stderr)
        else:
            print(f"t={time.perf_counter()-t1:.3f}s")
    else:
        print(f"Not rerun notebook {f} as interactive")
print("Failed:\n"+"\n".join(failed))
print("Interactive:\n"+"\n".join(i for i in interactive if interactive[i] is False))
print(f"Runtime: {time.perf_counter()-t0:.3f}s")


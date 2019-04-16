#!/usr/bin/env python3

import sys
import pymemcache.client
import h5py
import numpy
from pyFAI.io import get_isotime


def dump_h5(filename, server, port=11211):
    client = pymemcache.client.Client((server, port))
    stats = client.stats("slabs")
    slab_index = set(int(k.split(b":")[0]) for k in stats if b":" in k)
    print(slab_index)
    with h5py.File(filename, "w") as h5:
        entry = h5.require_group(get_isotime())
        for idx in slab_index:
            keys = stats.get(b"%i:cmd_set" % idx, 0)
            print("Slab #%s: %s keys" % (idx, keys))
            if keys:
                for key in client.stats("cachedump", str(idx), str(keys)):
                    ukey = key.decode()
                    print("    " + ukey)
                    entry[ukey] = numpy.frombuffer(client.get(key), dtype="uint8")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = "memcache_dump.h5"
    server = "127.0.0.1"
    print(filename, server)
    dump_h5(filename, server)

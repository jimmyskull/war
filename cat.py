import os
import pickle
import pprint
import sys
import zlib


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=1, width=240, compact=True)

    if len(sys.argv) == 1:
        obj = pickle.loads(zlib.decompress(sys.stdin.buffer.read()))
        pp.pprint(obj)
        sys.exit(0)

    for fid in sys.argv[1:]:
        strat, oid = fid.split('/')
        path = os.path.join('.war', strat, oid[:2], oid[2:])
        obj = pickle.loads(zlib.decompress(open(path, 'rb').read()))
        pp.pprint(obj)

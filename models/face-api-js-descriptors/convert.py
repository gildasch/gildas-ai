import json
import sys

def convert(filename):
    with open(filename, 'r') as f:
        s = f.read()
        d = eval(s)
        print(json.dumps(list(d.values())))

convert(sys.argv[1])

import ujson as json
import sys

def main():
    vec_file = sys.argv[1]
    pkl_file = '{}.json'.format('.'.join(vec_file.split('.')[:-1]))
    d = {}
    with open(vec_file, 'r') as f:
        for line in f.readlines():
            l = line.strip().split(' ')
            d[l[0]] = [float(x) for x in l[1:]]

    with open(pkl_file, 'w') as f:
        json.dump(d, f)

if __name__ == '__main__':
    main()

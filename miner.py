'''
Tool for mining image/audio search engines and storing in data directory for further processing
'''
import argparse

from mmfeat.miner import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', nargs=1, \
        help='type of engine to use', choices=['bing', 'google', 'freesound', 'flickr', 'imagenet'])
    parser.add_argument('query_file', nargs=1, \
        help='file that contains the list of queries (every line is a query)')
    parser.add_argument('data_dir', nargs=1, \
        help='location of directories')
    parser.add_argument('-n', '--num_files', type=int, action='store', \
        help='number of files to mine per query (default 20)', default=20)

    args = parser.parse_args()

    args.engine = args.engine[0]
    args.query_file = args.query_file[0]
    args.data_dir = args.data_dir[0]

    if args.engine == 'bing':
        miner = BingMiner(args.data_dir)
    elif args.engine == 'google':
        miner = GoogleMiner(args.data_dir)
    elif args.engine == 'freesound':
        miner = FreeSoundMiner(args.data_dir)
    elif args.engine == 'flickr':
        miner = FlickrMiner(args.data_dir)
    elif args.engine == 'imagenet':
        miner = ImageNetMiner(args.data_dir)

    queries = open(args.query_file).read().splitlines()
    miner.getResults(queries, args.num_files)
    miner.save()

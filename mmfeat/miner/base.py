'''
Base miner class
'''

import cPickle as pickle
import json
import os
import requests
import sys
import time
import urllib2
import yaml

class BaseMiner(object):
    def __init__(self, save_dir, config_path):
        '''
        save_dir:       the directory where we save files to
        config_path:    path to configuration file
        '''
        self.config = yaml.load(open(config_path))
        self.save_dir = save_dir
        self.cur_api_key = 0
        self.file_id = 1
        self.save_after_every_query = True
        self.sleep_time = 5
        self.max_sleep_time = 60*60 # 1 hour
        self.idx = {}

        if self.save_dir[-1] == '/':
            self.save_dir = self.save_dir[:-1] # strip trailing / if any

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif os.path.exists(self.save_dir + 'index.pkl'):
            self.idx        = pickle.load(open(self.save_dir + '/index.pkl', 'rb'))
            self.file_id    = max([int(fname.split('.')[0]) \
                                for sublist in self.idx.itervalues() \
                                for fname in sublist if fname is not None]) + 1

    def getResults(self, queries, limit):
        '''
        queries:    list of queries
        limit:      number of files per query
        '''
        self.results = {}
        for query in queries:
            if query in self.idx:
                n_stored_results = len(self.idx[query])
                if n_stored_results >= limit:
                    print query, 'already exists and has enough images, skipping..'
                    continue

            print 'Querying for %s' % query
            query_results = self.search(query, limit)
            if query in self.idx and n_stored_results >= len(query_results):
                print query, 'already has all available images, skipping..'
                continue

            self.results[query] = query_results

    def saveFile(self, result):
        '''
        result:     result object (BingResult, GoogleResult or FreesoundResult)
        '''
        if result.format in ['image/jpg', 'image/jpeg']:
            format = 'jpg'
        elif result.format in ['image/png']:
            format = 'png'
        elif result.format in ['image/gif']:
            format = 'gif'
        elif result.format in ['audio/ogg']:
            format = 'ogg'
        else: # unknown format, skipping
            return None

        if self.__engine__ == 'freesound':
            fname = '%s.%s' % (result.ID, format)
        else:
            fname = '%s.%s' % (self.file_id, format)
        path = '%s/%s' % (self.save_dir, fname)
        if self.__engine__ == 'freesound' and os.path.exists(path):
            print fname, ' - already exists'
            return

        # download the file
        f = urllib2.urlopen(result.url, timeout=20)
        if f is None:
            return None
        data = f.read()

        if self.__engine__ == 'freesound':
            response = json.loads(data)
            f = urllib2.urlopen(response['previews']['preview-hq-ogg'])
            if f is None:
                return None
            data = f.read()

        with open(path, 'wb') as fw:
            fw.write(data)

        print fname

        self.file_id += 1

        return fname

    def save(self):
        for query in self.results:
            if query not in self.idx:
                self.idx[query] = []

            print 'Saving %s files' % query

            for ii, result in enumerate(self.results[query]):
                fname = self.file_id
                try:
                    fname = self.saveFile(result)
                except:
                    print 'Fetching file for %s failed (url=%s, error=%s)' \
                        % (query, result.url, sys.exc_info()[1])
                    continue
                if fname is None: continue

                self.idx[query].append(fname)

                if self.save_after_every_query:
                    pickle.dump(self.idx, open('%s/index.pkl' % self.save_dir, 'wb'))

        pickle.dump(self.idx, open('%s/index.pkl' % self.save_dir, 'wb'))

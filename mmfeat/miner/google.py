'''
Google Custom Search API miner
'''

import requests
import time
import urllib2

from .base import BaseMiner

class GoogleResult(object):
    def __init__(self, result):
        self.ID     = None # Google has no unique IDs
        self.title  = result['title']
        self.url    = result['image']['thumbnailLink']
        self.format = 'image/png' # default thumbnail format
        # self.format = result['mime']

class GoogleMiner(BaseMiner):
    def __init__(self, save_dir):
        super(GoogleMiner, self).__init__(save_dir)
        self.__engine__ = 'google'

        self.search_id   = self.config['google']['search-id']
        self.api_keys   = self.config['google']['api-keys']
        self.lang       = self.config['google']['lang']
        self.format_url = 'https://www.googleapis.com/customsearch/v1?cx={}&key={}&q={}' \
                            + '&searchType=image&imgType=photo&start={}&lr={}'

    def getUrl(self, query, offset):
        query = urllib2.quote("'{}'".format(query))
        search_id = urllib2.quote(self.search_id)
        return self.format_url.format(search_id, self.api_keys[self.cur_api_key], query, \
            offset, self.lang)

    def _search(self, query, offset=1):
        url = self.getUrl(query, offset)
        r = requests.get(url)
        try:
            results = r.json()
        except ValueError:
            print('ERR: Request returned with code %s (%s)' % (r.status_code, r.text))
            if r.status_code == 429 or r.status_code == 503:
                print 'ERR (Google): Too many requests, sleeping..'
                time.sleep(self.sleep_time)
                self.sleep_time *= self.sleep_time
                if self.sleep_time > self.max_sleep_time:
                    self.sleep_time = 5
            else:
                time.sleep(self.sleep_time)

        total = int(results['searchInformation']['totalResults'])

        if 'items' not in results:
            return [], 0

        return [GoogleResult(res) for res in results['items']], total

    def search(self, query, limit=20):
        results, total = self._search(query, 1) # Google is 1-indexed
        while total > len(results) and len(results) < limit:
            more_results, _ = self._search(query, len(results))
            results += more_results
        return results

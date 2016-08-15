'''
ImageNet API miner
'''

import os
import requests
import shutil
import time
import urllib
import requests
import random

from .base import BaseMiner

try:
    from nltk.corpus import wordnet as wn
    from nltk.stem.wordnet import WordNetLemmatizer
except ImportError:
    import warnings
    warnings.warn('Could not find NLTK WordNet.')

class ImageNetResult(object):
    def __init__(self, url):
        self.url    = url
        self.format = 'image/jpg' # default format

class ImageNetMiner(BaseMiner):
    def __init__(self, save_dir, config_path='./miner.yaml'):
        super(ImageNetMiner, self).__init__(save_dir, config_path)
        self.__engine__ = 'imagenet'
        self.format_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'

        # maximum number of synsets to retrieve - we don't need all images necessarily, other-
        # wise we get enormous amounts of synsets for words like 'entity' or 'animal'
        self.max_synsets = 10000

        self.wnl = WordNetLemmatizer()

        # url cache
        self.imgnet_url_cache = {}

        # whether we "level up" in hierarchy if no images found
        self.level_up_if_no_images = True

    def getUrl(self, wnid):
        return self.format_url.format(wnid)

    def getResult(self, syn_id):
        url = self.getUrl(syn_id)
        r = requests.get(url)
        img_urls = r.text.splitlines()
        self.imgnet_url_cache[syn_id] = img_urls # add to cache
        return img_urls

    # get the image urls for a given synset
    def _search(self, synset):
        def wnid(synset):
            return '%s%.8d' % (synset.pos(), synset.offset())

        img_urls = []

        syn_id = wnid(synset)
        if syn_id in self.imgnet_url_cache:
            img_urls += self.imgnet_url_cache[syn_id]
        else:
            try:
                img_urls += self.getResult(syn_id)
            except ValueError, requests.exceptions.RequestException:
                if r.status_code == 104:
                    print 'Got a 104: sleeping for a bit and trying again...'
                    time.sleep(5) # just make sure ImageNet is not blocking us
                    try: # try again
                        img_urls += self.getResult(syn_id)
                    except:
                        return # ok, never mind - try a different synset

        return img_urls

    # get the image urls for a query
    def search(self, query, limit=20):
        if query[-2] == '-' and query[-1] in ['A', 'N', 'V']:
            word, pos = '-'.join(query.split('-')[:-1]), query.split('-')[-1]
        else:
            word, pos = query, None

        print 'Word', word, 'pos', pos

        # lemmatize and get synsets
        if pos == 'N':
            lem = self.wnl.lemmatize(word, pos=wn.NOUN)
            synsets = wn.synsets(lem, pos=wn.NOUN)
        elif pos == 'A':
            lem = self.wnl.lemmatize(word, pos=wn.ADJ)
            synsets = wn.synsets(lem, pos=wn.ADJ)
        elif pos == 'V':
            lem = self.wnl.lemmatize(word, pos=wn.VERB)
            synsets = wn.synsets(lem, pos=wn.VERB)
        else:
            lem = self.wnl.lemmatize(word)
            synsets = wn.synsets(lem)

        img_urls = []

        # helper function for getting hyponyms
        def get_hyponyms(synset, depth=0):
            hyponyms = set()
            depth += 1
            if depth == 40: # avoid maximum recursion exceeded errors
                return set()
            for hyponym in synset.hyponyms():
                hyponyms |= set(get_hyponyms(hyponym, depth))
            return hyponyms | set(synset.hyponyms())

        # 1. Get hyponyms to sample from
        for synset in synsets:
            img_urls += self._search(synset)
            for hn in get_hyponyms(synset):
                img_urls += self._search(hn)
                if len(img_urls) > self.max_synsets:
                    break

        # 2. If no images, try hyponyms of hypernyms
        if self.level_up_if_no_images and len(img_urls) == 0:
            for synset in synsets:
                for hypernym in synset.hypernyms():
                    img_urls += self._search(hypernym)
                    for hn in get_hyponyms(hypernym):
                        img_urls += self._search(hn)
                        if len(img_urls) > self.max_synsets:
                            break

        print 'Got ', len(img_urls), 'urls!'

        # 3. Sample n_images (i.e., limit) from the results
        random.shuffle(img_urls)
        results, cnt = [], 0
        for img_url in img_urls:
            if isinstance(img_url, basestring):
                # check that the URL exists
                try:
                    r = requests.head(img_url)
                    if r.status_code == requests.codes.ok:
                        results.append(ImageNetResult(img_url))
                        cnt += 1
                except requests.exceptions.RequestException:
                    continue
            if cnt == limit:
                break
        return results

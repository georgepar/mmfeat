'''
This demo shows how to get dispersion scores for a list of concepts and writes them
as a CSV file. See the paper:

Improving Multi-Modal Representations Using Image Dispersion: Why Less is Sometimes More
D. Kiela*, F. Hill*, A. Korhonen and S. Clark
Proceedings of ACL 2014, Baltimore, MA.
(*=equal contribution)

We obtain up to 20 images for the concepts in the list.
'''

import csv
import sys

sys.path.append('../..')
from mmfeat.miner import GoogleMiner
from mmfeat.bow import BoVW
from mmfeat.space import AggSpace

if __name__ == '__main__':
    concept_file =  './list_of_concepts.txt'
    data_dir =      './dispersion-images'
    output_file =   './dispersions.csv'
    n_images =      20

    #
    # Get a list of the concepts
    concepts = open(concept_file).read().splitlines()

    #
    # Obtain up to 20 images per concept from Google. This is similar to
    # calling the miner.py tool.
    print('Mining concept images')
    miner = GoogleMiner(data_dir, '../../miner.yaml')
    miner.getResults(concepts, n_images)
    miner.save()

    #
    # Learn BoVW representations for the images. You can also use CNN here.
    print('Loading data and getting representations')
    model = BoVW(100)
    model.load(data_dir)
    model.fit()

    #
    # Build a lookup for the concepts
    print('Building lookup')
    lkp = model.toLookup()

    #
    # Turn into a visual space and get the dispersion scores
    print('Loading visual space')
    vs = AggSpace(lkp, 'mean')
    vs.getDispersions(rescale=False)

    #
    # Write dispersions to a CSV file
    with open(output_file, 'wb') as csvfp:
        csvw = csv.writer(csvfp, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for concept, dispersion in vs.dispersions.items():
            csvw.writerow([concept, dispersion])

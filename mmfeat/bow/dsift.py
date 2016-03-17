'''
DSIFT Feature Extractor

Based on Yangqing Jia's excellent code at

https://github.com/Yangqing/dsift-python
'''

import numpy as np
from scipy.ndimage import filters

class DsiftExtractor(object):
    def __init__(self):
        # sift features
        self.gS = 8
        self.pS = 16
        self.nrml_thres = 1.0
        self.sigma = 0.8
        self.sift_thres = 0.2
        self.Nangles = 8
        self.Nbins = 4
        self.Nsamples = self.Nbins**2
        self.alpha = 9.0
        self.angles = np.array(range(self.Nangles))*2.0*np.pi/self.Nangles

        self.GH, self.GW = self.gen_dgauss(self.sigma)

        # compute the weight contribution map
        # weights is the contribution of each pixel to the corresponding bin center
        sample_res = self.pS / np.double(self.Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,self.Nbins*2,2)) / 2.0 / self.Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        self.weights = weights_h * weights_w

    def gen_dgauss(self, sigma):
        '''
        generating a derivative of Gauss filter on both the X and Y direction.
        '''
        fwid = np.int(2*np.ceil(sigma))
        G = np.array(range(-fwid,fwid+1))**2
        G = G.reshape((G.size,1)) + G
        G = np.exp(- G / 2.0 / sigma / sigma)
        G /= np.sum(G)
        GH,GW = np.gradient(G)
        GH *= 2.0/np.sum(np.abs(GH))
        GW *= 2.0/np.sum(np.abs(GW))
        return GH,GW

    def process_image(self, image):
        '''
        image: a M*N image which is a numpy 2D array. If you pass a color image,
            it will automatically be converted to a grayscale image.

        Return values:
        feaArr: the feature array, each row is a SIFT feature
        '''

        image = image.astype(np.double)
        if image.max() > 1: # map to [0,1]
            image /= 255;
        if image.ndim == 3: # map color image to 2d
            image = np.mean(image, axis=2)

        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = remH/2
        offsetW = remW/2
        rangeH = np.arange(offsetH, H-pS+1, gS)
        rangeW = np.arange(offsetW, W-pS+1, gS)
        feat = self.calculate_sift_grid(image, rangeH, rangeW)
        feat = self.normalize_sift(feat)

        # return [Npoints x Dim] array
        return feat.reshape(-1, feat.shape[-1])

    def calculate_sift_grid(self, image, rangeH, rangeW):
        H, W = image.shape
        feat = np.zeros((len(rangeH), len(rangeW), self.Nsamples*self.Nangles))
        IH = filters.convolve(image, self.GH, mode='nearest')
        IW = filters.convolve(image, self.GW, mode='nearest')
        I_mag = np.sqrt(IH ** 2 + IW ** 2)
        I_theta = np.arctan2(IH, IW)
        I_orient = np.empty((H, W, self.Nangles))
        for i in range(self.Nangles):
            I_orient[:,:,i] = I_mag * np.maximum(
                    np.cos(I_theta - self.angles[i]) ** self.alpha, 0)
        for i, hs in enumerate(rangeH):
            for j, ws in enumerate(rangeW):
                feat[i, j] = np.dot(self.weights,
                                    I_orient[hs:hs+self.pS, ws:ws+self.pS]\
                                        .reshape(self.pS**2, self.Nangles)
                                   ).flat
        return feat

    def normalize_sift(self, feat):
        siftlen = np.sqrt(np.sum(feat**2, axis=-1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feat /= siftlen[:, :, np.newaxis]
        # suppress large gradients
        feat[feat > self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feat[hcontrast] /= np.sqrt(np.sum(feat[hcontrast]**2, axis=-1))\
                [:, np.newaxis]
        return feat

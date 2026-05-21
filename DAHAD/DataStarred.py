from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, TwoSlopeNorm
import pandas as pd 
import pickle
#from photutils.detection import DAOStarFinder, find_peaks
#from photutils.aperture import CircularAperture
#from photutils.centroids import centroid_sources, centroid_2dg, centroid_com
#import pyregion


from scipy.ndimage import binary_dilation



#tabla1 = pd.read_csv(Path(__file__).resolve().parent / "suportdata" / "tablea1.csv")

class ImageFits:
    
    def __init__(self, path_data,path_weight=None):
        path_file = Path(path_data)
        #path_weight = Path(path_weight)
        
        if not path_file.is_file():
            raise FileNotFoundError(f"File not found: {path_file}")
        # if not path_weight.is_file():
        #     raise FileNotFoundError(f"File weight not found: {path_weight}")

        self.path_file = path_file
        #self.path_weight = path_weight
        self._readfits()
        #self._readweight()
        #self.found_object()
        #filter object 
       
        
        
    def _readfits(self):
        self.header0 = fits.open(self.path_file)[0].header
        data_raw, self.header = fits.getdata(self.path_file, header=True)
        self.FILTER  = self.header0.get("FILTER")
        #this keys works for JWST
        self.TARGNAME = self.header0.get("TARGPROP",self.header0.get("TARGNAME"))
        self.RA =  self.header0.get("TARG_RA",self.header0.get("RA_TARG"))
        self.DEC =  self.header0.get("TARG_DEC",self.header0.get("DEC_TARG"))
        
        self.EXPTIME =  self.header0.get("EXPTIME")
        self.wcs = WCS(self.header)
        self.data_raw = data_raw.astype(float)
        # self.mean, self.median, self.std = sigma_clipped_stats(data_raw, sigma=5.0)
        # self.data_raw = data_raw - self.median
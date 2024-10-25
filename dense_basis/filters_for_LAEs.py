#imports

import numpy as np
import os.path as op
import os
import glob

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique, vstack, join
from astropy.io import fits
from astropy.io import ascii, fits

import tkinter as tk
from tkinter import *
import tkinter.messagebox
import tkinter.font as font

import random

import pandas as pd

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe

import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astroML.plotting import scatter_contour
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.cosmology import Planck15 as cosmo

from scipy.stats import gaussian_kde
import scipy.stats as st
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

import statistics

from IPython.display import display, HTML

# import pyregion

from tqdm.auto import tqdm

import statsmodels.api as sm

from PIL import Image, ImageOps

import seaborn as sns

import csv

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from IPython.display import Image as Images

# from __future__ import print_function
# from dustmaps.config import config
# config['data_dir'] = "/Users/nicolefirestone/opt/anaconda3/lib/python3.9/site-packages/dustmaps/data/sfd"
#if the line below throws an error, reinstall dustmaps (pip install dustmaps) 
#because it requires specific versions of other installations
# import dustmaps.sfd
# dustmaps.sfd.fetch()
# from dustmaps.sfd import SFDQuery

import time

import sys
#from IPython.display import Audio
#sound_file = '/Users/nicolefirestone/Downloads/chimes.mp3'

from astropy.table import Table # astropy.table knows what a .fits file is

import scipy.integrate as it

import re
from scipy.signal import find_peaks

def get_eff_wavelength_new(filter_transmission):
    filt_transmission_data = get_pkg_data_filename('/Users/nicolefirestone/Desktop/grd_scl/dense_basis/dense_basis/build/lib/dense_basis/filters/' + filter_transmission)

    filt_wav = (np.array([x.split()[0] for x in open(filt_transmission_data).readlines()])).astype(float)
    filt = (np.array([x.split()[1] for x in open(filt_transmission_data).readlines()])).astype(float)

    integrand1_filt = filt_wav*filt #1.25 airmasses of atmospheric attenuation already taken into account in transmission file 
    integrand2_filt = filt
    filt_eff = it.trapz(integrand1_filt,filt_wav)/it.trapz(integrand2_filt,filt_wav)
    
    #get filter name 
    string = (filter_transmission).astype(str)
#     head1, sep1, tail1 = string.partition('/') 
#     head2, sep2, tail2 = tail1.partition('/')
#     head3, sep3, tail3 = tail2.partition('.')
#     filter_name = head3  
    filter_name = (((((string.partition('/'))[2]).partition('/'))[2]).partition('.'))[0]
#     print(filter_name, 'EFFECTIVE WAVELENGTH is: ', round(filt_eff,2))
    
#     globals()['%s' % filter_name + "eff"] = filt_eff
    
    return filt_eff

def get_FWHM(a_filter):
    filt_transmission_data = get_pkg_data_filename('/Users/nicolefirestone/Desktop/grd_scl/dense_basis/dense_basis/build/lib/dense_basis/filters/' + a_filter)

    wavelength = (np.array([x.split()[0] for x in open(filt_transmission_data).readlines()])).astype(float)
    transmission = (np.array([x.split()[1] for x in open(filt_transmission_data).readlines()])).astype(float)

    d = transmission - (max(transmission) / 2)
#     plt.plot(transmission, d)
    index = np.where(d > 0)[0]
#     print(index)
    
#     figure = plt.figure()
#     plt.plot(wavelength, transmission)
#     plt.vlines(wavelength[index[-1]], 0, max(transmission))
#     plt.vlines(wavelength[index[0]], 0, max(transmission))
    
    return wavelength[index[-1]], wavelength[index[0]]

def get_filter_name(filter_data):
    string = (filter_data).astype(str)
    filter_name = (((((string.partition('/'))[2]).partition('/'))[2]).partition('.'))[0]
    
    return filter_name

def choose_filters_laes(filter_list, filt_dir, z):

  filter_data = np.genfromtxt((filt_dir+filter_list), skip_header=0,skip_footer=0, names=None, dtype=None, delimiter=' ')
  
  filter_data_orig = []
  for i in filter_data:
      a_filter_data = (i).astype(str)
      filter_data_orig.append(a_filter_data)
      
  wave_lya = (1 + z)*1216
  print("z = ", z)
  print("lya wavelength = ", round(wave_lya,2), "A\n\n")
  
  FWHM_max_thresh = 400 #A #threshold for medium band is 300 A --> https://en.wikipedia.org/wiki/Photometric_system
  
  ###############
  
  filters_use = []
  
  all_filts_names = []
  
  filters_use_bool = []
  
  eff_waves = []
  
  for a_filter in filter_data_orig:
      print(get_filter_name(a_filter))
      all_filts_names.append(get_filter_name(a_filter))
      FWHM_max, FWHM_min = get_FWHM(a_filter)
      print("upper FWHM:", FWHM_max, "A")
      print("lower FWHM:",FWHM_min, "A")
      FWHM = FWHM_max - FWHM_min
      print("FWHM:", round(FWHM, 2), "A")
      eff_wave = get_eff_wavelength_new(a_filter)
      eff_waves.append(eff_wave)
      print("eff_wave:", round(eff_wave, 2))
      if ((((wave_lya > FWHM_max) or (wave_lya < FWHM_min)) or (FWHM > FWHM_max_thresh)) and (get_filter_name(a_filter) != "f43_IRAC_ch4_2" and get_filter_name(a_filter) != "f42_IRAC_ch3_2")):
  #     if ((get_filter_name(a_filter) != "f43_IRAC_ch4_2" and get_filter_name(a_filter) != "f42_IRAC_ch3_2")):
          filters_use.append(a_filter)
          print("--> used\n")
          filters_use_bool.append(True)
      else:
          print("--> omitted\n")
          filters_use_bool.append(False)
  
  omitted_filts = list(set(filter_data_orig)-set(filters_use))
  omitted_filts_names = []
  
  if len(omitted_filts) == 0:
      print(len(omitted_filts), "filters have been omitted")
  
  if len(omitted_filts) == 1: 
      print(len(omitted_filts), "filter has been omitted:")
      for filt in omitted_filts:
          print(get_filter_name(filt), round(get_eff_wavelength_new(filt), 2))
          omitted_filts_names.append(get_filter_name(filt))
          
  if len(omitted_filts) > 1: 
      print(len(omitted_filts), "filters have been omitted:")
      for filt in omitted_filts:
          print(get_filter_name(filt), round(get_eff_wavelength_new(filt), 2))
          omitted_filts_names.append(get_filter_name(filt))
  
  return filters_use_bool

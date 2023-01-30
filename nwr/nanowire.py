#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:57:48 2018
Nanowire class definition.

@author: dwinge
"""

# This is the nanowire class. Future versions can include:
# 1. Modes for the given wire
# 2.

import numpy as np
from scipy import interpolate

class Nanowire:
    
    index_dir = '/home/dwinge/Codes/NanowireResonances/index/'
    index_static = 1
    index_lamb = np.array # converted to nm
    index_nk = np.array
    index_unit = 'nm'
        
    def __init__(self, diameter, material=None, index=3.3+0.0j, scaleindex=1000.):
        """
        Creates an nanowire object.

        Parameters
        ----------
        diameter : float
            Nanowire diameter in nm.
        material : str, optional
            Material string matching a database file. The default is None.
        index : float, optional
            Constant index if no material is supplied. The default is 3.3+0.0j.
        scaleindex : float, optional
            Tweaking variable, set to 1 if importing nm based refractive index. 
            The default is 1000 and means importing um based refractive index.

        Returns
        -------
        None.

        """
        self.diameter = diameter
        self.material = material
        self.index = index
        
        if self.material :
            if self.read_index(material,scale=scaleindex) :
                self.index_static = 0
                self.f = interpolate.interp1d(self.index_lamb,self.index_nk)
                
       
    def read_index(self,in_file,scale=1000.): # scale from um to nm
        """Read refractive index file."""
        suffixes = ['.txt'] # Fixme: We could loop over these in the future
        try:
            with open( self.index_dir+in_file+suffixes[0] ) as f :
                index_data = np.loadtxt( f )
                self.index_lamb = index_data[:,0]*scale 
                # The following construction saves memory
                self.index_nk = 1j*index_data[:,2]  # imag
                self.index_nk += index_data[:,1]    # real
                return True
        except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
            print("The requested index file was not found in",self.index_dir)
            return False
            
    def get_index(self,lamb) :
        """Calculates the index at given wavelength"""
        if self.index_static == 1 :
            return self.index
        else:
            return self.f(lamb)
        
    def get_size(self,lamb) :
        """ Calculates the size parameter for a given wavelength"""
        k= 2.*np.pi/lamb # assume outer medium index=1 
        return k*self.diameter/2. # size parameter k*r
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:02:35 2018
This module hols the ananlysis part. The most important feature is that it 
keeps track of the variables that we change during a calculation, and the 
ones that are kept fixed.

@author: dwinge
"""

import numpy as np
import nwr.expcoeff_cyl as expcyl

class Analysis:
    
    # analysis variables, can be set from outside 
    # (not sure if one can actually declare like this)
    lamb = np.array
    zeta = np.array
    phi = np.array
    coeff = np.array
    T = np.array
    Qsca = np.array
    Qext = np.array
    
    def __init__(self,_nw,lamb,zeta,phi,out_index=1.0,maxn=10):
        """
        Generate an analysis object that holds the results. Run analysis using
        the run_analysis method.

        Parameters
        ----------
        _nw : nanowire object
            Instance of the nanowire class.
        lamb : array or float
            wavelengths to study
        zeta : array or float
            incoming angle w r t nanowire axis
        phi : array or float
            scattering angle w r t plane of incidence
        out_index : float, optional
            surrounding refr. index. The default is 1.0.
        maxn : int, optional
            maximum order of coefficients. The default is 10.

        Returns
        -------
        None.

        """
        self._nw = _nw
        self.lamb = lamb
        self.zeta = zeta
        self.phi = phi
        self.out_index = out_index
        self.maxn = maxn

    def run_analysis(self,maxorder=None) :
        """
        Runs the main analysis in the correct order.

        Parameters
        ----------
        maxorder : int, optional
            maximum order of coefficients. The default is None.

        Returns
        -------
        array of floats
            2 x 2 scattering matrices T as Nwavel x Nzeta x Nphi x 4
            ordered as [[T1, T4],[T3, T2]]
        array of floats
            Scattering coefficients Qsca, as Nwavel x Nzeta x 2
            where the last index is for parallel (0) and perpedicular (1) polarization
        array of floats
            Extinction coefficients Qext, similar to Qsca

        """
        if maxorder is None:
            maxorder=self.maxn
        self.calc_coeff()
        self.calc_T(maxorder)
        self.calc_Q()
        return self.T, self.Qsca, self.Qext
        
    def calc_coeff(self) :
        """Calculate the a and b coefficients of Bohren and Huffman Eqs (8.29-8.31)"""

        # Now we call abccoeff() to give us our coefficients        
        Nl, Nx = len(np.atleast_1d(self.lamb)),len(np.atleast_1d(self.zeta))
        self.coeff = np.zeros((Nl,Nx,self.maxn+1,4),dtype=complex) # Four different coefficients
        for l in np.arange(0,Nl) :
            for m in np.arange(0,Nx) :
                tmp = expcyl.abcoeff(self._nw,
                                      np.atleast_1d(self.lamb)[l],
                                      np.atleast_1d(self.zeta)[m],
                                      self.out_index)
                
                # We need to run a few checks on the raw data
                topn = len(tmp[0])
                ceiln = min(topn,self.maxn+1)
                # Now enter the data into our structure
                for i in np.arange(0,4) :
                    self.coeff[l,m,:ceiln,i] = tmp[i][:ceiln]
                

    def calc_T(self,maxorder=None) :
        """Calculate the scattering matrix T of Bohren and Huffman Eq. (8.34)"""
        if maxorder is None:
            maxorder=self.maxn
        # Now we use the result to generate the T-matrices
        Np = len(np.atleast_1d(self.phi))
        Nl, Nx = len(np.atleast_1d(self.lamb)),len(np.atleast_1d(self.zeta))
        self.T = np.zeros((Nl,Nx,Np,4),dtype=complex)
        nvec = np.arange(1,maxorder+1)
        # Zero order contribution
        self.T[:,:,:,0] = self.coeff[:,:,0,2,None]
        self.T[:,:,:,1] = self.coeff[:,:,0,1,None]
        # Perform whole ananlysis
        for p in np.arange(0,Np) :
            Theta = np.pi-np.pi*np.atleast_1d(self.phi)[p]/180.
            self.T[:,:,p,0] = self.T[:,:,p,0] + 2.*np.dot(self.coeff[:,:,1:maxorder+1,2],
                                                          np.cos(nvec*Theta))
            self.T[:,:,p,1] = self.T[:,:,p,1] + 2.*np.dot(self.coeff[:,:,1:maxorder+1,1],
                                                          np.cos(nvec*Theta))
            self.T[:,:,p,2] = self.T[:,:,p,2] - 1j*2.*np.dot(self.coeff[:,:,1:maxorder+1,0],
                                                             np.sin(nvec*Theta))
        self.T[:,:,:,3] = -self.T[:,:,:,2]
        
    def calc_Q(self, maxorder=None) :
        """Caclulate the extinction and scattering coefficients"""
        if maxorder is None:
            # This is not really implemented yet
            maxorder=self.maxn
        # Now we use the result to generate the T-matrices
        Nl, Nx = len(np.atleast_1d(self.lamb)),len(np.atleast_1d(self.zeta))
        # Q is a real parameter and takes size, zeta and wavelength as input
        # We distinguish case I and case II and scatt and extinction
        self.Qsca, self.Qext = np.zeros((Nl,Nx,2)), np.zeros((Nl,Nx,2))
        # We calculate in two ways, 
        # 1. Loop over lambda and zeta and do vector operation
        size = self._nw.get_size(self.lamb)
        for l in np.arange(0,Nl) :
            for m in np.arange(0,Nx) :
                # Case I
                self.Qsca[l,m,0] = 2./size[l] * (abs(self.coeff[l,m,0,2])**2+
                                                 2*(np.vdot(self.coeff[l,m,1:,2],self.coeff[l,m,1:,2])+
                                                    np.vdot(self.coeff[l,m,1:,0],self.coeff[l,m,1:,0])).real)
                self.Qext[l,m,0] = 2./size[l] * (self.coeff[l,m,0,2]+2*self.coeff[l,m,1:,2].sum()).real
                
                # Case II
                self.Qsca[l,m,1] = 2./size[l] * (abs(self.coeff[l,m,0,1])**2+
                                                 2*(np.vdot(self.coeff[l,m,1:,1],self.coeff[l,m,1:,1])+
                                                    np.vdot(self.coeff[l,m,1:,3],self.coeff[l,m,1:,3])).real)
                self.Qext[l,m,1] = 2./size[l] * (self.coeff[l,m,0,1]+2*self.coeff[l,m,1:,1].sum()).real
            
    
    def check_Q(self,theta_zero_idx) :
        """Extra method to verify the results of the calc_Q method"""
        # Dimensions of T
        Nl, Nx = len(np.atleast_1d(self.lamb)),len(np.atleast_1d(self.zeta)) 
        check_ext = np.zeros((Nl,Nx,2))
        # Get the size parameter
        size = self._nw.get_size(self.lamb)
        # Pick out theta=0
        for l in np.arange(0,Nl) :
            # Case I
            check_ext[l,:,0] = 2./size[l] * self.T[l,:,theta_zero_idx,0].real
            # Case II
            check_ext[l,:,1] = 2./size[l] * self.T[l,:,theta_zero_idx,1].real
            
        return check_ext
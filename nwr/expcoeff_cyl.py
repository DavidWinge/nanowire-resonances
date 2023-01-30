#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module for calculating scattering parameters for infinite cylinders
Created on Tue Nov 13 09:22:26 2018

The calculations are based on the book of Bohren and Huffman [1]:
    [1] Bohren, C. F. and Huffman, D. R., Absorption and scattering of 
        light by small particles, Wiley-Interscience, New York, 1998.
        
@author: dwinge
""" 
    
import numpy as np
import scipy.special as spec
#import nwr.nanowire as nw

def abcoeff(_nw, lamb, zeta, out_index) :
    """ Translates objects into variables relevant for calculation."""
    in_index = _nw.get_index(lamb)
    k= 2.*np.pi/lamb*out_index # wavenumber in outer medium 
    x = k*_nw.diameter/2. # size parameter k*r
    m = in_index/out_index # relative refractive index, in general complex
    
    return expcoeff_cyl( x,m,zeta )  

def expcoeff_cyl( x, m, zeta=90, hk=1, conv=1 ) :
    """Calculates the expansion coefficients for the infinite circular cylinder
    
    Parameters
    ----------
    x : float
        size parameter
    m : float
        relative index (inside over outside cylinder)
    zeta : float, optional (default is 90)
        incidence angle (degrees) relative to cylinder principle angle 
    hk : integer, 1 or 2 (default is 1)
        Hankel function kind
    conv : float, optional (default is 1.0)
        additional convergence criteria
        
    Returns
    -------
    anp,ann,bnp,bnn : numpy arrays
        expansion coefficients
    """


    zeta = zeta/180*np.pi     # inclination angle in radians

    # Calculate truncation number
    xsin = x*np.sin(zeta)
    M = np.ceil(conv*(xsin + 4*(xsin**(1./3)) + 2))
    n = np.arange(0,M+1)

    # Calculate auxiliary variables
    xi = x*np.sin(zeta)
    eta = x*np.sqrt(m**2 - np.cos(zeta)**2)
    jneta = spec.jv(n, eta)
    djneta = spec.jvp(n, eta)
    jnxi = spec.jv(n, xi)
    djnxi = spec.jvp(n, xi)
    if (hk == 1) :
        hnxi = spec.hankel1(n,xi)
        dhnxi = spec.h1vp(n,xi)
    elif (hk ==2) :
        hnxi = spec.hankel2(n,xi)
        dhnxi = spec.h2vp(n,xi)
    else :
        print("Error in expcoeff_cyl.py: Hankel function kind not 1 or 2.")
        print("Exiting.")
        exit()
     

    An = 1j*xi*(xi*djneta*jnxi - eta*jneta*djnxi)
    Bn = xi*(m**2*xi*djneta*jnxi - eta*jneta*djnxi)
    Cn = n*np.cos(zeta)*eta*jneta*jnxi*(xi**2/eta**2 - 1)
    Dn = n*np.cos(zeta)*eta*jneta*hnxi*(xi**2/eta**2 - 1)
    Vn = xi*(m**2*xi*djneta*hnxi - eta*jneta*dhnxi)
    Wn = 1j*xi*(eta*jneta*dhnxi - xi*djneta*hnxi)
    
    wvd = (Wn*Vn + 1j*Dn**2)
    cd = 1j*Cn*Dn
    
    idx = np.isnan(wvd)
    
    ## Calculate expansion coefficients
    anp = (Cn*Vn - Bn*Dn)/wvd
    ann = -(An*Vn - cd)/wvd
    bnp = (Wn*Bn + cd)/wvd
    bnn = -1j*(Cn*Wn + An*Dn)/wvd
    
    anp[idx] = 0.
    ann[idx] = 0.
    bnp[idx] = 0.
    bnn[idx] = 0.
    
    return anp, ann, bnp, bnn
    

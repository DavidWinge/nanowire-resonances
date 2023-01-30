#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:34:25 2018
Modified on Wed Dec 12 and added to nwr package
This module holds functionality to plot the modes in an infinite cylinder.
@author: dwinge
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec

def MNn(n,k,h,z,rho) :
    """
    Calculates the M and N cylindrical vector harmonics of Bohren and Huffman, 
    unlabeled but following Eqs. (8.28)

    Parameters
    ----------
    n : int
        order
    k : float
        wavevector number in inv nm
    h : float
        separation constant
    z : array
        z coordinates
    rho : array
        effective radius, rho=r*sqrt(k**2-h**2)

    Returns
    -------
    Mn : complex array
        vector harmonics M ordered as r, theta
    Nn : complex array
        vector harmonics N ordered as r, theta, z

    """
    Z = spec.jv(n,rho)
    Zp= spec.jvp(n,rho)
    first = n*Z/rho
    third = np.sqrt(k**2-h**2)*Z
    
    Mn = np.zeros((len(rho),3),dtype=complex)
    Nn = np.zeros((len(rho),3),dtype=complex)
    
    Mn[:,0] = np.sqrt(k**2-h**2)*(1j*first)*np.exp(1j*h*z)
    Mn[:,1] = np.sqrt(k**2-h**2)*(-Zp)*np.exp(1j*h*z)
    Nn[:,0] = np.sqrt(k**2-h**2)/k*(1j*h*Zp)*np.exp(1j*h*z)
    Nn[:,1] = np.sqrt(k**2-h**2)/k*(-h*n*Z/rho)*np.exp(1j*h*z)
    Nn[:,2] = np.sqrt(k**2-h**2)/k*(third)*np.exp(1j*h*z)
    
    return Mn,Nn

def print_mode(my_nw,zeta,wavel,my_n,rho_fac=1.0,rho_limit=1.0,my_dpi=300,
               phase_shift=0,index_out=1.0,cdir='',saveplots=False) : 
    """
    Prnt the modes at 

    Parameters
    ----------
    my_nw : object
        Nanowire class instance.
    zeta : float
        incidence angle (rad) with respect to nanowire axis
    wavel : float
        wavelength in nm.
    my_n : int
        order.
    rho_fac : float, optional
        Factor of maximum rho to use for plot. The default is 1.0.
    rho_limit : float, optional
        Lower limit of outer rho for quiver plot. The default is 1.0.
    my_dpi : int, optional
        Saved image resolution. The default is 300.
    phase_shift : float, optional
        Extra phase shift in radians. The default is 0.
    index_out : float, optional
        Surrounding refractive index. The default is 1.0.
    cdir : str, optional
        Plot save directory. The default is ''.
    saveplots : bool, optional
        Save plots. The default is False.

    Returns
    -------
    None.

    """
    # Specify system
    R = my_nw.diameter # radius in nm
    k = 2.*np.pi/wavel # wavenumber in inv nm
    m = my_nw.get_index(wavel)/index_out    # relative refractive index
    h = -k*np.cos(zeta) # 
    
    # Need to calculate the vector harmonics in component form in r,phi,theta
    start_rho=rho_fac*k*R*np.sqrt(m**2-np.cos(zeta)**2)*1e-3
    end_rho = rho_fac*k*R*np.sqrt(m**2-np.cos(zeta)**2)*1
    edge = k*R*np.sqrt(m**2-np.cos(zeta)**2)
    
    # Dense calculation space
    my_rho = np.linspace(start_rho,end_rho,num=50)
    my_phi = np.linspace(0.0,2.*np.pi,num=100)
    
    # Sparse calculation space with a reduced rho- and phi-vectors
    new_rho = np.linspace(start_rho,end_rho,num=8)
    new_phi = np.linspace(0.,2.*np.pi,num=25)
    
    # Zero-valued array for fields in r,theta and z direction
    Mn, Nn = np.zeros((len(my_rho),3),dtype=complex), np.zeros((len(my_rho),3),dtype=complex)
    Mn_s, Nn_s = np.zeros((len(new_rho),3),dtype=complex), np.zeros((len(new_rho),3),dtype=complex)
    
    # New meshgrid for visualization
    x1, z1 = np.meshgrid(my_phi, my_rho)
    
    # --------------------------------------------------------------
    # Calculation starts here 
    # --------------------------------------------------------------
    
    
    # Short for the phase factors
    phase = np.exp(1j*(my_n*my_phi+phase_shift))
    new_phase = np.exp(1j*(my_n*new_phi+phase_shift))
    
    # Calculate the vector fields for the given parameters
    Mn, Nn = MNn(my_n,k,h,0,my_rho)
    Mn_s, Nn_s = MNn(my_n,k,h,0,new_rho)
    
    # -------------------------------------------------------------
    # Plot M-type
    # -------------------------------------------------------------
    
    # Plotting the length of the field vectors in polar coordinates
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    
    # Draw the contour
    ax.contourf(my_phi,np.real(my_rho),np.sqrt( np.real(np.outer(Mn[:,0],phase))**2+np.real(np.outer(Mn[:,1],phase))**2),cmap='jet',zorder=0)
    ax.plot(my_phi,np.ones_like(my_phi)*np.real(edge),'r-')
    # Add a quiver plot to visualize the field lines
    short_idxs = np.where( new_rho > rho_limit )
    
    Mnx = np.outer(Mn_s[short_idxs,0],np.cos(new_phi)*new_phase)-np.outer(Mn_s[short_idxs,1],np.sin(new_phi)*new_phase)
    Mny = np.outer(Mn_s[short_idxs,0],np.sin(new_phi)*new_phase)+np.outer(Mn_s[short_idxs,1],np.cos(new_phi)*new_phase)
    
    # Normalize to longest vector
    Mnorm = np.real(np.amax(Mnx*Mnx.conjugate()+Mny*Mny.conjugate()))
    # Draw arrows
    ax.quiver(new_phi,np.real(new_rho[short_idxs]), np.real(Mnx)/Mnorm, np.real(Mny)/Mnorm, pivot='middle',units='xy')
    # Remove the contour labels
    ax.set_yticklabels([])
    ax.set_title('Vector harmonic {0} order {1:d}'.format('M',my_n))
    # Save figure
    if saveplots :
        plt.savefig(cdir+'fields{0}{1:d}.png'.format('M',my_n),dpi=my_dpi)
    plt.show()
    
    # -------------------------------------------------------------
    # Plot N-type
    # -------------------------------------------------------------
    
    # Plotting the length of the field vectors in polar coordinates
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    
    # Draw the contour
    ax.contourf(my_phi,np.real(my_rho),np.sqrt( np.real(np.outer(Nn[:,0],phase))**2+np.real(np.outer(Nn[:,1],phase))**2+np.real(np.outer(Nn[:,2],phase))**2),cmap='jet',zorder=0)
    ax.plot(my_phi,np.ones_like(my_phi)*np.real(edge),'r-')
    # Add a quiver plot to visualize the field lines
    short_idxs = np.where( new_rho > rho_limit )
    
    Nnx = np.outer(Nn_s[short_idxs,0],np.cos(new_phi)*new_phase)-np.outer(Nn_s[short_idxs,1],np.sin(new_phi)*new_phase)
    Nny = np.outer(Nn_s[short_idxs,0],np.sin(new_phi)*new_phase)+np.outer(Nn_s[short_idxs,1],np.cos(new_phi)*new_phase)
    
    # Normalize to longest vector
    Nnorm = np.real(np.amax(Nnx*Nnx.conjugate()+Nny*Nny.conjugate()))
    # Draw arrows
    ax.quiver(new_phi,np.real(new_rho[short_idxs]), np.real(Nnx)/Mnorm, np.real(Nny)/Nnorm, pivot='middle',units='xy')
    # Remove the contour labels
    ax.set_yticklabels([])
    ax.set_title('Vector harmonic {0} order {1:d}'.format('N',my_n))
    # Save figure
    if saveplots :
        plt.savefig(cdir+'fields{0}{1:d}.png'.format('N',my_n),dpi=my_dpi)
    plt.show()
    
    # -------------------------------------------------------------
    # Plot N-type longitudinal component
    # -------------------------------------------------------------
    
    # Plotting the length of the field vectors in polar coordinates
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    
    # Draw the contour
    ax.contourf(my_phi,np.real(my_rho),np.real(np.outer(Nn[:,2],phase)),cmap='jet',zorder=0)
    ax.plot(my_phi,np.ones_like(my_phi)*np.real(edge),'r-')
    ax.set_yticklabels([])
    ax.set_title('Vector harmonic {0} z-component order {1:d}'.format('N',my_n))
    # Save figure
    if saveplots :
        plt.savefig(cdir+'fields{0}{1:d}.png'.format('Nz',my_n),dpi=my_dpi)
    plt.show()
    

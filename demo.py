# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Demonstration of the nwr package for Mie theory calculations

# %% [markdown]
# As a first example, we study the polarization anisotropy of set of nanowires with different radius at UV wavelengths. The nwr package comes with a database of refractive indices for different materials. Please check these data before doing anything serious with the package, like papers etc. 

# %%
# Standard numeric python packages
import numpy as np
import matplotlib.pyplot as plt
# Import the nanowire class and the analysis class
import nwr.nanowire as nw
import nwr.analysis as ana

# %%
# First we create a set of nanowires 
diameters = [10., 25., 50., 100.0, 200.0] # create python list
my_nws = [] # create empty list

# Use the database of materials for a quick look at InP wires
for d in diameters:
    my_nws.append(nw.Nanowire(d,material='InP'))

# Setup a range of interesting wavelengths
nlamb = 691
lamb_min,lamb_max = 210,900
lin_lamb = np.linspace(lamb_min,lamb_max,num=nlamb,endpoint=True)

# %% [markdown]
# Now we have the nanowires and the interesting wavelengths. To study the absorption extinction coefficient, we need to supply the zeta array and the phi array (in degrees), which are the angles of the incoming radiation with respect to the nanowire axis, and the scattering angle with respect to the plane of the incoming light, respectively. See Fig. 8.3 of Bohren and Huffman for a geometrical guide. 
# In our case, we use perpendicular illumination and thus a single value zeta=90 and phi=180, but it is possible to supply an array. Then we do the calculations as detailed by Bohren and Huffman (2008).

# %%
zeta = 90.0
# Create analysis objects for each nanowire
my_analysis = []
for w in my_nws:
    my_analysis.append(ana.Analysis(w,lin_lamb,zeta,phi=180,maxn=10))
    # Calculate the a's and b's of Bohren and Huffman eqs (8.29, 8.31), 
    # the amplitude scattering matrix elements of Bohren and Huffman eq (8.34),
    # and the extinction coefficients, Bohren and Huffman eqs (8.36,8.37) (can be done step step by calling
    # calc_coeff(), calc_T() and calc_Q() in that particular order.
    my_analysis[-1].run_analysis()


# %% [markdown]
# The info is now contained in objects contained in the analysis class. Let's have a look. The absorption efficiency is defined as the extinction efficiency minus the scattering efficiency, see Bohren & Huffman eqs (8.36,8.37). In the resulting plots below we can see the extreme polarization anisotropy at smaller wavelengths. 

# %%
fig, ax = plt.subplots()

# The two polarization directions are included in the calculated Q's
pol = ['para', 'perp']
style = ['-','--']
for case in range(0,2) :
    for k, a in enumerate(my_analysis):
        # Q is 3D array and structured like wavelength, zeta, and case (0 is paralell, 1 is perpendicular)
        ax.plot(lin_lamb,a.Qext[:,0,case]-a.Qsca[:,0,case],style[case],label=f'R={diameters[k]/2:.0f} {pol[case]}')
    
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorption efficiency')
ax.set_xlim(200,500)
plt.legend()
plt.show()

# Create also a plot with the ratio of absorption, abs(90)/abs(0)
fig, ax = plt.subplots()

for k, a in enumerate(my_analysis):
    # Q is 3D array and structured like wavelength, zeta, and case (0 is paralell, 1 is perpendicular)
    ax.plot(lin_lamb,(a.Qext[:,0,1]-a.Qsca[:,0,1])/(a.Qext[:,0,0]-a.Qsca[:,0,0]),label=f'R={diameters[k]/2:.0f} perp/para')
    
ax.plot(lin_lamb,np.ones_like(lin_lamb),'k--')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorption efficiency ratio')
ax.set_xlim(200,500)
plt.legend()
plt.show()

# %% [markdown]
# Let's say that we are searching for a resonance in our wavelength range. Then it might be instructive to look at the a,b coefficients of Bohren and Huffman eqs. (8.29-8.31) separately. This is done in the following snippet for the n and p case, where p is parallel and n is non-paralell (perpendicular) polarization, respectively. The results for the case R=50 nm are plotted.

# %%
# We plot orders 0:3
f, axarr= plt.subplots(4,4)#, sharex='col',sharey='row')
ncols, nrows = 4,4

# choose zeta (only one zeta in current analysis)
zeta_idx = 0

# labels
label={0:'anp - T3',1:'ann - T2',2:'bnp - T1',3:'bnn - T4'}

my_ana = my_analysis[3] # diameter of 100 nm, radius 50 nm

for m in np.arange(0,nrows) :
    for n in np.arange(0,ncols) :
        axarr[m,n].plot(lin_lamb,abs(my_ana.coeff[:,zeta_idx,m,n]),'k--')
        axarr[m,n].plot(lin_lamb,my_ana.coeff[:,zeta_idx,m,n].real,'b-')
        axarr[m,n].plot(lin_lamb,my_ana.coeff[:,zeta_idx,m,n].imag,'r-')
        axarr[m,n].set_xlim(lamb_min,lamb_max)
        
for n in np.arange(0,ncols) :
    axarr[ncols-1,n].set_xlabel('wavelength [nm]')
    axarr[0,n].set_title(label[n])
    axarr[n,0].set_ylabel('Order {0:d}'.format(n))

figure = plt.gcf() # get current figure
figure.set_size_inches(19, 10)

# %% [markdown]
# In the plot above for R=50 nm it seems like something interesting is happening at 500 nn wavelength. It possible using another module "modes" to show the modes inside a nanowire for different wavelengths. It is important to know that what is plotted are the real quantities of feilds that are in general complex.
# What we see is that there are spiraling modes of both electronic and magnetic character at this particular wavelength.

# %%
import nwr.modes as modes
# Takes input: nanowire object, zeta(rad), wavel(nm), order(int)
# As one of the options, specify rho_fac relative to R for plot extent
modes.print_mode(my_nws[3],np.pi/2,500,1,rho_fac=1.5)


# %%

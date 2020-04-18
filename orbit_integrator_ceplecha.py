#!/usr/bin/env python


"""
=============== Meteoroid Orbit Determination ===============
Created on Tues Jul 7 2:51:10 2015
@author: Trent Jansen-Sturgeon

Inputs: The initial ECEF position (m), ECEF velocity (m/s), and time (jd).
Outputs: The meteoroid's orbital elements about the Sun, or Earth (a,e,i,omega,OMEGA,theta).
         - using J2000 fundamental epoch
"""

import numpy as np
from astropy import units as u
from numpy.linalg import norm

from trajectory_utilities import ECEF2ECI, ECI2HCI, PosVel2OrbitalElements
from orbital_utilities import  OrbitObject, PlotOrbit3D, generate_ephemeris

# Define some constants
mu_e = 3.986005000e14 #4418e14 # Earth's standard gravitational parameter (m3/s2)
mu_s = 1.32712440018e20  # Sun's standard gravitational parameter (m3/s2)
# mu_j = 1.26686534e17  # Jupiter's standard gravitational parameter (m3/s2)
mu_m = 4.9048695e12  # Moon's standard gravitational parameter (m3/s2)
R_e = 6371.0e3  # Earth's mean radius (m)
R_s = 695.5e6  # Sun's mean radius (m)
w_e = 7.2921158553e-5  # Earth's rotation rate (rad/s)
J2 = 0.00108263  # Zonal Harmonic Perturbation
a_earth = 1.495978875e11  # Earth's semi-major axis around the sun (m)
AU = 149597870700.0  # One Astronomical Unit (m)
SMA_JUPITER = 5.20336301 * u.au


def propagateOrbit(stateVec, Plot=False, Sol='NoSol', verbose=True, ephem=False):
    '''
    The CelpechaOrbit function calculates the origin of the meteor by correcting the 
    magnitude and direction of the velocity vector to exclude Earth's influencing
    effects.
    '''
    
    Pos_geo = stateVec.position
    Vel_geo = stateVec.vel_xyz
    t0 = stateVec.epoch
    
    ''' Convert from geo to ECI coordinates '''
    [Pos_ECI, Vel_ECI] = ECEF2ECI(Pos_geo, Vel_geo, t0)

    ''' Calculate V_g, the geocentric velocity outside Earth's influence '''
    V_inf = norm(Vel_ECI)
    print('V inf : {0:.1f}'.format(V_inf))
    V_esc = np.sqrt(2 * mu_e / norm(Pos_ECI))
    V_g = np.sqrt(V_inf**2 - V_esc**2)

    ''' Calculate the zenith attraction due to Earth's influence '''
    # Calculate the velocity vector in the ENU frame
    ra = float(np.arctan2(Pos_ECI[1], Pos_ECI[0]))
    dec = float(np.arcsin(Pos_ECI[2] / norm(Pos_ECI)))
    C_ENU2ECI=np.array([[-np.sin(ra),-np.sin(dec)*np.cos(ra),np.cos(dec)*np.cos(ra)],
                        [ np.cos(ra),-np.sin(dec)*np.sin(ra),np.cos(dec)*np.sin(ra)],
                        [      0    ,       np.cos(dec)     ,       np.sin(dec)    ]])
                        
    Vel_ENU = np.dot(C_ENU2ECI.T, Vel_ECI)
    
    # Calculate the azimuth and zenith angles
    a_c = np.arctan2(-Vel_ENU[0], -Vel_ENU[1])
    z_c = np.arccos(-Vel_ENU[2] / V_inf)
    
    # Make zenith correction due to Earth's influence
    dz_c = 2 * np.arctan((V_inf-V_g) * np.tan(z_c / 2) / (V_inf + V_g))
    z_g = z_c + dz_c
    a_g = a_c

    ''' Combine the zenith and velocity corrections '''
    Vel_ENU = V_g * np.vstack((-np.sin(z_g) * np.sin(a_g),
        -np.sin(z_g) * np.cos(a_g), -np.cos(z_g)))
    Vel_ECI = np.dot(C_ENU2ECI, Vel_ENU)

    ''' Convert from ECI to HCI coordinates '''
    [Pos_HCI, Vel_HCI] = ECI2HCI(Pos_ECI, Vel_ECI, t0)
    
    ra_corr = float(np.arctan2(-Vel_ECI[1], -Vel_ECI[0]))
    dec_corr = float(np.arcsin(-Vel_ECI[2] / norm(Vel_ECI)))
    
    COE = PosVel2OrbitalElements(Pos_HCI, Vel_HCI, 'Sun', 'Classical')
    DeterminedOrbit = OrbitObject('Heliocentric',
                                COE[0,-1] * u.m, 
                                COE[1,-1],
                                COE[2,-1] * u.rad,
                                COE[3,-1] * u.rad,
                                COE[4,-1] * u.rad,
                                COE[5,-1] * u.rad,
                                ra_corr * u.rad,
                                dec_corr * u.rad,
                                V_g * u.m / u.second)
    
    # Plots if required
    if Plot:
        # Plot the orbit in 3D
        PlotOrbit3D([DeterminedOrbit], t0, Sol)

    ephem_dict = None
    if ephem: # Also return the ephemeris dict
        t_max = np.argmin(t); t_jd = t[:t_max]
        pos_hci = Pos_HCI[:,:t_max]
        ephem_dict = generate_ephemeris(pos_hci, t_jd)
    
    return DeterminedOrbit, ephem_dict
    
#    # How Ceplecha calculates the zenith angle - same result
#    ra=np.arctan2(Vel_ECI[1],Vel_ECI[0])
#    dec=np.arcsin(Vel_ECI[2]/norm(Vel_ECI))
#    lon=np.arctan2(Pos_ECI[1],Pos_ECI[0])
#    lat=np.arcsin(Pos_ECI[2]/norm(Pos_ECI))
#    z_c=np.arccos(np.sin(dec)*np.sin(lat)+np.cos(dec)*np.cos(lat)*np.cos(lon-ra))

#%%============================================================================
#%%


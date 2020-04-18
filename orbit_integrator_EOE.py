#!/usr/bin/env python



"""
=============== Meteoroid Orbit Determination ===============
Created on Tues Jul 7 2:51:10 2015
@author: Trent Jansen-Sturgeon

Inputs: The initial ECEF position (m), ECEF velocity (m/s), and time (jd).
Outputs: The meteoroid's orbital elements about the Sun, or Earth (a,e,i,omega,OMEGA,theta).
         - using J2000 fundamental epoch
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from numpy.linalg import norm
from astropy.time import Time
from astropy.coordinates import get_body, HCRS

from orbital_utilities import ThirdBodyPerturbation, NRLMSISE_00, PlotPerts, \
    PlotIntStep, OrbitObject, PlotOrbit3D, PlotOrbitalElements, generate_ephemeris
from trajectory_utilities import ECEF2ECI, ECI2HCI, HCI2ECI, EarthPosition, \
    HCRS2HCI, PosVel2OrbitalElements, OrbitalElements2PosVel
from atm_functions import dragcoef, reynolds, knudsen


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

Pert = []

def SunDynamics(y, t, Perturbations, record=False):
    '''
    The state rate dynamics are used in Runge-Kutta integration method to 
    calculate the next set of equinoctial element values.
    '''

    ''' State Rates '''
    # Equinoctial element vector decomposed
    [p, f, g, h, k, L] = y.flatten()[:6]

    # Short-hand elements
    w = 1 + f * np.cos(L) + g * np.sin(L)
    s2 = 1 + h**2 + k**2
    alpha2 = h**2 - k**2
    pm = np.sqrt(p / mu_s) / w
    hk = h * np.sin(L) - k * np.cos(L)

    # State rate constant
    b = np.vstack((0, 0, 0, 0, 0, np.sqrt(mu_s * p) * (w / p)**2))

    Backward = True
    if Perturbations >= 10:
        Backward = False
        Perturbations = Perturbations - 10

    if Backward:

        # State rate matrix
        A = np.array([[        0          ,          2 * p * pm           ,           0            ],
                      [ pm * w * np.sin(L), pm * ((w + 1) * np.cos(L) + f),     -pm * g * hk       ],
                      [-pm * w * np.cos(L), pm * ((w + 1) * np.sin(L) + g),      pm * f * hk       ],
                      [        0          ,              0                , pm * s2 * np.cos(L) / 2],
                      [        0          ,              0                , pm * s2 * np.sin(L) / 2],
                      [        0          ,              0                ,        pm * hk         ]])

        # Calculate the current position of meteoroid (m)
        r = p / w
        Pos_HCI = r / s2 * np.vstack((np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L),
                                      np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L),
                                      2 * hk))
        Vel_HCI = -1 / (s2 * pm * w) * np.vstack((np.sin(L) + alpha2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha2 * g,
                                                  -np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha2 * f,
                                                  -2 * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)))

        # Body frame unit vectors
        X_body = Pos_HCI / r
        Z_body = np.cross(Pos_HCI, Vel_HCI, axis=0) / norm(np.cross(Pos_HCI, Vel_HCI, axis=0), axis=0)
        Y_body = np.cross(Z_body, X_body, axis=0) / norm(np.cross(Z_body, X_body, axis=0), axis=0)

        # Rotation matrix
        HCI2Body = np.vstack((X_body.T, Y_body.T, Z_body.T))

        ''' Secondary Body Perturbations: Earth '''
        # Position vector from Sun to Earth (m)
        rho_e = EarthPosition(t)

        # Third body perturbation acceleration (Earth) -HCI frame
        u_earth = ThirdBodyPerturbation(Pos_HCI, rho_e, mu_e)

        if Perturbations:
            ''' Secondary Body Perturbations: Moon '''
            # Position vector from Earth to Moon (m)
            T = Time(t, format='jd', scale='utc')
            Moon_GCRS_SC = get_body('moon',T)
            rho_m = np.vstack(Moon_GCRS_SC.transform_to(HCRS(obstime=T)).
                    cartesian.xyz.to(u.meter).value)
            rho_m = HCRS2HCI(rho_m)

            # Third body perturbation acceleration (Moon) -HCI frame
            u_moon = ThirdBodyPerturbation(Pos_HCI, rho_m, mu_m)

            ''' Total Perturbing Acceleration (in the body frame) '''
            u_tot = np.dot(HCI2Body, u_earth + u_moon)

        else:
            ''' Total Perturbing Acceleration (in the body frame) '''
            u_tot = np.dot(HCI2Body, u_earth); rho_m = np.nan
        
        if record:
            
            EARTH = mu_e / (norm(Pos_HCI - rho_e))**2
            MOON = mu_m / (norm(Pos_HCI - rho_m))**2
            SUN = mu_s / r**2
            
            global Pert
            Pert.append([t, EARTH, MOON, SUN, np.nan, np.nan])

        ''' State Rate Equation '''
        y_dot = np.vstack(( np.dot(A, u_tot) + b, 0, 0))

    else:
        ''' State Rate Equation '''
        y_dot = np.vstack(( b, 0, 0))

    return y_dot

def EarthDynamics(y, t, Perturbations, record=False):
    '''
    The state rate dynamics are used in Runge-Kutta integration method to 
    calculate the next set of equinoctial element values.
    '''

    ''' State Rates '''
    # Equinoctial element vector decomposed
    [p, f, g, h, k, L] = y.flatten()[:6]

    # Short-hand elements
    w = 1 + f * np.cos(L) + g * np.sin(L)
    s2 = 1 + h**2 + k**2
    alpha2 = h**2 - k**2
    pm = np.sqrt(p / mu_e) / w
    hk = h * np.sin(L) - k * np.cos(L)

    # State rate constant
    b = np.vstack((0, 0, 0, 0, 0, np.sqrt(mu_e * p) * (w / p)**2))

    # State rate matrix
    A = np.array([[        0          ,          2 * p * pm           ,           0            ],
                  [ pm * w * np.sin(L), pm * ((w + 1) * np.cos(L) + f),     -pm * g * hk       ],
                  [-pm * w * np.cos(L), pm * ((w + 1) * np.sin(L) + g),      pm * f * hk       ],
                  [        0          ,              0                , pm * s2 * np.cos(L) / 2],
                  [        0          ,              0                , pm * s2 * np.sin(L) / 2],
                  [        0          ,              0                ,        pm * hk         ]])

    # Position in ECI coordinates
    r = p / w  # Radius (m)
    Pos_ECI = r / s2 * np.vstack((np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L),
                                  np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L),
                                  2 * hk))
    Vel_ECI = -1 / (s2 * pm * w) * np.vstack((np.sin(L) + alpha2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha2 * g,
                                              -np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha2 * f,
                                              -2 * (h * np.cos(L) + k * np.sin(L) + f * h + g * k)))

    # Body frame unit vectors (r=x_body,t=y_body,n=z_body)
    X_body = Pos_ECI / r
    Z_body = np.cross(Pos_ECI, Vel_ECI, axis=0) / norm(np.cross(Pos_ECI, Vel_ECI, axis=0), axis=0)
    Y_body = np.cross(Z_body, X_body, axis=0) / norm(np.cross(Z_body, X_body, axis=0), axis=0)

    # Rotation matrix
    ECI2Body = np.vstack((X_body.T, Y_body.T, Z_body.T))

    ''' Third Body Perturbations: Sun '''
    # Position vector from Earth to the Sun (m)
    T = Time(t, format='jd', scale='utc')
    sun_GCRS = get_body('sun',T)
    rho_s = np.vstack(sun_GCRS.cartesian.xyz.to(u.meter).value)

    # Third body perturbation acceleration (Sun) -ECI frame
    u_sun = ThirdBodyPerturbation(Pos_ECI, rho_s, mu_s)

    if Perturbations:
        ''' Atmospheric Drag Perturbation '''
        [M, CA] = y.flatten()[6:]
        if M > 0 and CA > 0 and r < R_e + 10e6: # To the top of the exosphere
            
            # NRLMSISE-00 atmospheric model
            [temp, atm_pres, rho_a, sos, dyn_vis] = NRLMSISE_00(Pos_ECI, t)
            
            # Atmospheric velocity
            v_atm = np.cross(np.vstack((0, 0, w_e)), Pos_ECI, axis=0)
        
            # Velocity relative to the atmosphere
            v_rel = Vel_ECI - v_atm
            v = norm(v_rel, axis=0)
            
            # Constants for drag coeff
            mach = v / sos # Mach Number
            re = reynolds(vel=v, rho=rho_a, dvisc=dyn_vis, length=2 * np.sqrt(CA / np.pi)) # Reynolds Number
            kn = knudsen(mach=mach, re=re) # Knudsen Number
            Cd = dragcoef(re=re, mach=mach, kn=kn, A=2.0) # Drag Coefficient
            # Cd = 2.0 # Approximation
        
            # Total drag perturbation
            u_drag = -1.0 / (2 * M) * rho_a * Cd * CA * v * v_rel  
            
        else:
            # Total drag perturbation
            u_drag = np.nan

        ''' Gravitational J2 Perturbation '''
        # Gravitational perturbation components
        J2_r = -(3 * mu_e * J2 * R_e**2) / (2 * r**4) * (1 - 12 * hk**2 / s2**2)
        J2_t = -(12 * mu_e * J2 * R_e**2) / (r**4) * hk * (h * np.cos(L) + k * np.sin(L)) / s2**2
        J2_n = -(6 * mu_e * J2 * R_e**2) / (r**4) * (2 - s2) * hk / s2**2

        u_J2 = np.vstack((J2_r, J2_t, J2_n))

        ''' Third Body Perturbations: Moon '''
        # Position vector from Earth to Moon (m)
        moon_GCRS = get_body('moon',T)
        rho_m = np.vstack(moon_GCRS.cartesian.xyz.to(u.meter).value)

        # Third body perturbation acceleration (Moon) -ECI frame
        u_moon = ThirdBodyPerturbation(Pos_ECI, rho_m, mu_m)

        ''' Total Perturbing Acceleration '''
        if np.isnan(norm(u_drag)):
            u_tot = u_J2 + np.dot(ECI2Body, u_moon + u_sun)
        else:
            u_tot = u_J2 + np.dot(ECI2Body, u_moon + u_sun + u_drag)
        
        if record:
            
            EARTH = mu_e / r**2
            MOON = mu_m / (norm(Pos_ECI-rho_m))**2
            SUN = mu_s / (norm(Pos_ECI-rho_s))**2
            
            global Pert
            Pert.append([t, EARTH, MOON, SUN, norm(u_J2), norm(u_drag)])
    else:
        u_tot = np.dot(ECI2Body, u_sun)

    ''' State Rate Equation '''
    y_dot = np.vstack(( np.dot(A, u_tot) + b, 0, 0))

    return y_dot


def ODE45(f, Y0, t0, dt0, Perturbations=True):
    
    # Constants (change to suit)
    Min_dt = 0.001 / (24*60*60) # Min dt is 0.001 sec
    Max_dt = 6.0 / 24 # Max dt is 6 hrs
    eps = 1e-3 # 1e-4 is too small and doesn't finish
    
    if f == EarthDynamics:
        OrbitBody = 'Earth'
    elif f == SunDynamics:
        OrbitBody = 'Sun'
    else:
        print( f, 'is not valid function: ODE45')
        exit()
        
    Continue_loop = True
    while Continue_loop:
        
        # Conversions and constants
        step = dt0 * 24 * 60 * 60
        
        # 4th order Runge Kutta
        y1 = Y0
        k1 = step * f(y1, t0, Perturbations, True)
        
        y2 = Y0 + 1./5 * k1
        k2 = step * f(y2, t0 + 1./5 * dt0, Perturbations)
        
        y3 = Y0 + 3./40 * k1 + 9./40 * k2
        k3 = step * f(y3, t0 + 3./10 * dt0, Perturbations)
        
        y4 = Y0 + 44./45 * k1 - 56./15 * k2 + 32./9 * k3
        k4 = step * f(y4, t0 + 4./5 * dt0, Perturbations)
        
        y5 = Y0 + 19372./6561 * k1 - 25360./2187 * k2 + 64448./6561 * k3 - 212./729 * k4
        k5 = step * f(y5, t0 + 8./9 * dt0, Perturbations)
        
        y6 = Y0 + 9017./3168 * k1 - 355./33 * k2 + 46732./5247 * k3 + 49./176 * k4 - 5103./18656 * k5
        k6 = step * f(y6, t0 + dt0, Perturbations)
        
        Y = Y0 + 35./384 * k1 + 500./1113 * k3 + 125./192 * k4 - 2187./6784 * k5 + 11./84 * k6
        
        # 5th order Runge Kutta
        k7 = step * f(Y, t0 + dt0, Perturbations)
        Z = Y0 + 5179./57600 * k1  + 7571./16695 * k3 + 393./640 * k4 - 92097./339200 * k5 + 187./2100 * k6 + 1./40 * k7
        
        # Error estimator
        Pos_Y = OrbitalElements2PosVel(Y, OrbitBody, 'Equinoctial')[0]
        Pos_Z = OrbitalElements2PosVel(Z, OrbitBody, 'Equinoctial')[0]
                
        error = norm(Pos_Z - Pos_Y)
    
        # Optimise time step
        if error != 0:
            s = (eps / error)**0.2
        else:
            s = 2.
        
        # If dt is too big/small, decrease/increase it within bounds
        if s >= 10./9 and abs(dt0) < Max_dt:
            dt = 10./9 * dt0
            Continue_loop = False
        elif s < 1 and abs(dt0) > Min_dt:
            dt0 = 9./10 * dt0
            global Pert; Pert = Pert[:-1] # Delete last perturbation record
        else:
            dt = dt0
            Continue_loop = False
    
    # Update the time to correspond with Z    
    t = t0 + dt0
            
    return Z, t, dt

def propagateOrbit(stateVec, Perturbations=True, Plot=False, Sol='NoSol', verbose=True, ephem=False):
    '''
    The propagateOrbit function calculates the origin of the meteor by reverse
    propergating the position and velocity out of the atmosphere using the
    equinoctial element model.
    '''
    import time
    starttime = time.time()
    
    SharedGlobals('Reset')

    Pos_geo = stateVec.position
    Vel_geo = stateVec.vel_xyz
    t0 = stateVec.epoch
    M = stateVec.mass
    CA = stateVec.cs_area

    ''' Convert from geo to ECI coordinates '''
    [Pos_ECI, Vel_ECI] = ECEF2ECI(Pos_geo, Vel_geo, t0)
    
    ''' Calculate the initial meteors equinoctial elements in ECI coords '''
    EOE = PosVel2OrbitalElements(Pos_ECI, Vel_ECI, 'Earth', 'Equinoctial')

    ''' Numerically integrate until its beyond the Earth's SOI '''
    # Earth's levels of satellites
    R_ATM = 500.0e3 + R_e
    Above_ATM = False
    R_LEO = 2000.0e3 + R_e
    Above_LEO = False
    R_GEO = 35786.0e3 + R_e
    Above_GEO = False
    R_MOON = 384400.0e3
    Above_MOON = False

    # Earth's sphere of influence (SOI)
    R_SOI = a_earth * (mu_e / mu_s)**0.4  # ~3*R_MOON

    # Setup initial values
    Y = np.vstack((EOE, M, CA))
    w = 1 + Y[1] * np.cos(Y[5]) + Y[2] * np.sin(Y[5])
    r = Y[0] / w  # Initial radius
    rmax = r # Maximum radius reached
    OrbitType = 'Heliocentric'  # Initial assumption
    
    # Check the time step is negative
    dt = -0.1 / (24 * 60 * 60)  # 1sec (JD)
    t = np.array([t0])  # Start the time at epoch (JD)

    # Integrate with RK4 until outside Earth's sphere of influence
    if verbose:
        print('Rewinding from Bright Flight...')
    while r < R_SOI:

        # Runge Kutta Integration (Dormand-Prince)
        [Y_new, t_new, dt_new] = ODE45(EarthDynamics, Y[:, -1:], t[-1], dt, Perturbations)
        
        # Save EOE's and time steps
        Y = np.hstack((Y, Y_new))
        t = np.hstack((t, t_new))
        dt = dt_new

        # Calculate the current distance from Earth (m)
        w = 1 + Y_new[1] * np.cos(Y_new[5]) + Y_new[2] * np.sin(Y_new[5])
        r = Y_new[0] / w
        
        # Record the maximum radius reached
        if r > rmax: rmax = r
        
        # Print progress
        sys.stdout.write('\r%.3f%%' % (r / (10 * R_SOI) * 100))
        sys.stdout.flush()

        if r > R_ATM and Above_ATM == False:
            if verbose:
                print('\rPassing ATM.')
            Above_ATM = True

        if r > R_LEO and Above_LEO == False:
            if verbose:
                print('\rPassing LEO.')
            Above_LEO = True

        if r > R_GEO and Above_GEO == False:
            if verbose:
                print('\rNow passing GEO!')
            Above_GEO = True

        if r > R_MOON and Above_MOON == False:
            if verbose:
                print('\rWe\'re over the Moon!!')
            Above_MOON = True

        if r < rmax / 2.: # Make sure it isn't coming back
            OrbitType = 'Geocentric'
            break
    
    t_soi = [t[-1]]
#    t_soi = [t[-1],  # Save time for plotting reference
##        Time('2010-06-13T13:35:00.0',format='isot',scale='utc').jd,
#        Time('2010-06-09T06:04:00.0',format='isot',scale='utc').jd]

    if OrbitType == 'Geocentric':

        # Convert to geocentric orbital elements
        [Pos_ECI, Vel_ECI] = OrbitalElements2PosVel(Y, 'Earth', 'Equinoctial')
        COE = PosVel2OrbitalElements(Pos_ECI, Vel_ECI, 'Earth', 'Classical')
        ra_corr = np.nan; dec_corr = np.nan; v_g = np.nan

    elif OrbitType == 'Heliocentric':
        if verbose:
            print('\rNow entering a Sun centred orbit...')

        # Convert the equinoctial elements to position and velocity
        [Pos_ECI, Vel_ECI] = OrbitalElements2PosVel(Y, 'Earth', 'Equinoctial')

        # Convert the ECI position and velocity to HCI coordinates
        [Pos_HCI, Vel_HCI] = ECI2HCI(Pos_ECI, Vel_ECI, t)

        ''' Numerically integrate until we get 10*R_SOI and back again '''
        # Convert to heliocentric orbital elements
        EOE = PosVel2OrbitalElements(Pos_HCI, Vel_HCI, 'Sun', 'Equinoctial')
        Y = np.vstack((EOE, np.vstack((M, CA)) * np.ones((1, np.shape(EOE)[1]))))

        # Integrate with RK4 until outside 10 times the sphere of influence
        ReachTenSOI = False
        # t0 = Time('2010-06-09T06:04:00.0', format='isot', scale='utc').jd # Time of TCM4
        while t0 != t[-1]:
            
            # Runge Kutta Integration (Dormand-Prince)
            [Y_new, t_new, dt_new] = ODE45(SunDynamics, Y[:, -1:], t[-1], dt, Perturbations)
            
            # Save EOE's and time steps
            Y = np.hstack((Y, Y_new))
            t = np.hstack((t, t_new))
            dt = dt_new
            
            # Record the current HCI position and velocity
            [Pos_hci, Vel_hci] = OrbitalElements2PosVel(Y_new, 'Sun', 'Equinoctial')
            Pos_HCI = np.hstack((Pos_HCI, Pos_hci))
            Vel_HCI = np.hstack((Vel_HCI, Vel_hci))

            # Calculate the current distance from Earth
            r = norm(EarthPosition(t[-1]) - Pos_hci)
            
            # Print progress
            sys.stdout.write('\r%.3f%%' % (r / (10 * R_SOI) * 100))
            sys.stdout.flush()
            
            if r > 10 * R_SOI and ReachTenSOI == False:
                if verbose:
                    print('\rWe reached 10*R_SOI, now we go... \nBack to the future!!!')
                ReachTenSOI = True
                dt = abs(dt)  # Ensure the time step is now positive
                Perturbations = Perturbations + 10  # Encode the time step info
            
            if abs(t0 - t[-1]) < abs(dt):  # This ensures we end intergration on epoch
                dt = (t0 - t[-1])
            
            if abs(t[-1] - t0) > 3272.7:  # This is 3272.7 days of flight time (max)
                print('\rExceeded maximum flight time:')
                print('Did not escape 10 times Earth\'s sphere of influence.\n')
                break

        COE = PosVel2OrbitalElements(Pos_HCI, Vel_HCI, 'Sun', 'Classical')
        [Pos_eci, Vel_eci] = HCI2ECI(Pos_hci, Vel_hci, t[-1])
        ra_corr = float(np.arctan2(-Vel_eci[1], -Vel_eci[0]))
        dec_corr = float(np.arcsin(-Vel_eci[2] / norm(Vel_eci)))
        v_g = norm(Vel_eci)

        if COE[0, -1] < 0 and COE[1, -1] > 1:
            OrbitType = 'Hyperbolic'

    print('\r' + OrbitType + ' orbit determined!')
    if verbose:
        print("--- %s seconds ---" % (time.time() - starttime))

    # Create the orbit object
    DeterminedOrbit = OrbitObject(OrbitType,
                                  COE[0, -1] * u.m,
                                  COE[1, -1],
                                  COE[2, -1] * u.rad,
                                  COE[3, -1] * u.rad,
                                  COE[4, -1] * u.rad,
                                  COE[5, -1] * u.rad,
                                  ra_corr * u.rad,
                                  dec_corr * u.rad,
                                  v_g * u.m / u.second)
                                  
    # Plots if required
    if Plot:
        if len(Pert) > 0:
            # Plot the perturbations over time
            # global Pert
            PlotPerts(Pert)
               
        # Plot dt
        PlotIntStep(t)
        
        # Plot the orbit in 3D
        PlotOrbit3D([DeterminedOrbit], t0, Sol)
        
        # Plot the individual orbital elements
        PlotOrbitalElements(COE, t, t_soi, Sol)


    ephem_dict = None
    if ephem: # Also return the ephemeris dict
        t_max = np.argmin(t); t_jd = t[:t_max]
        pos_hci = Pos_HCI[:,:t_max]
        ephem_dict = generate_ephemeris(pos_hci, t_jd)
    
#    from scipy.integrate import simps
#    global Pert#, time_tot, pert_tot
#    time_tot = (t.max() - t.min()) * u.d
#    if (Perturbations - 10) == True:
#        pert_tot = (-24*60*60 * simps(Pert[1,1:] + Pert[0,1:], t[1:Pert.shape[1]])) * (u.m / u.second)
#    else:
#        pert_tot = 0
    
    return DeterminedOrbit, ephem_dict#, time_tot, pert_tot

def SharedGlobals(Action):

    if Action == 'Get':
        global time_tot, pert_tot
        return time_tot, pert_tot
    elif Action == 'Reset':
        global Pert
        Pert = []


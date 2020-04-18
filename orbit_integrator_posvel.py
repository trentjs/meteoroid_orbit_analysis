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
from astropy.coordinates import get_body, GCRS

from trajectory_utilities import ECEF2ECI, ECI2HCI, HCI2ECI, PosVel2OrbitalElements
from orbital_utilities import ThirdBodyPerturbation, NRLMSISE_00, PlotPerts, \
    PlotIntStep, OrbitObject, PlotOrbit3D, PlotOrbitalElements, generate_ephemeris
from atm_functions import dragcoef, reynolds, knudsen


# Define some constants
mu_e = 3.986005000e14 #4418e14 # Earth's standard gravitational parameter (m3/s2)
mu_s = 1.32712440018e20  # Sun's standard gravitational parameter (m3/s2)
mu_m = 4.9048695e12  # Moon's standard gravitational parameter (m3/s2)
R_e = 6371.0e3  # Earth's mean radius (m)
R_s = 695.5e6  # Sun's mean radius (m)
w_e = 7.2921158553e-5  # Earth's rotation rate (rad/s)
J2 = 0.00108263  # Zonal Harmonic Perturbation
a_earth = 1.495978875e11  # Earth's semi-major axis around the sun (m)
AU = 149597870700.0  # One Astronomical Unit (m)

# Earth's sphere of influence (SOI)
R_SOI = a_earth * (mu_e / mu_s)**0.4  # ~3*R_MOON

def SunDynamics(t, X, t0):
    '''
    The state rate dynamics are used in Runge-Kutta integration method to 
    calculate the next set of equinoctial element values.
    '''

    ''' State Rates '''
    # State parameter vector decomposed
    # t = np.array(t)/(24*60*60) + t0
    Pos_HCI = np.vstack((X[:3])); Vel_HCI = np.vstack((X[3:]))

    ''' Primary Gravitational Acceleration '''
    a_grav = - mu_s * Pos_HCI / norm(Pos_HCI)**3

    ''' Total Perturbing Acceleration (in the inertial frame) '''
    a_tot = a_grav
        
    ''' State Rate Equation '''
    X_dot = np.vstack((Vel_HCI, a_tot))

    return X_dot.flatten()

def EarthDynamics(t, X, params):
    '''
    The state rate dynamics are used in Runge-Kutta integration method to 
    calculate the next set of equinoctial element values.
    '''

    ''' State Rates '''
    # State parameter vector decomposed
    Pos_ECI = np.vstack((X[:3])); Vel_ECI = np.vstack((X[3:6]))
    M = X[6]; CA = X[7]; r = norm(Pos_ECI)
    [t0, Perturbations] = params
    
    ''' Primary Gravitational Acceleration '''
    a_grav = - mu_e * Pos_ECI / r**3

    ''' Third Body Perturbations: Sun '''
    # Position vector from Earth to the Sun (m)
    T = Time(t / (24*60*60) + t0, format='jd', scale='utc')
    sun_gcrs = get_body('sun',T)
    rho_s = np.vstack(sun_gcrs.transform_to(GCRS(obstime=T)).cartesian.xyz.to(u.meter).value)

    # Third body perturbation acceleration (Sun) -ECI frame
    a_sun = ThirdBodyPerturbation(Pos_ECI, rho_s, mu_s)
    

    if Perturbations:

        ''' Atmospheric Drag Perturbation - Better Model Needed '''
        if M > 0 and CA > 0 and r < R_e + 10e6: # To the top of the exosphere
            
            # NRLMSISE-00 atmospheric model
            [temp, atm_pres, rho_a, sos, dyn_vis] = NRLMSISE_00(Pos_ECI, T)
            
            # Atmospheric velocity
            v_atm = np.cross(np.vstack((0, 0, w_e)), Pos_ECI, axis=0)
        
            # Velocity relative to the atmosphere
            v_rel = Vel_ECI - v_atm
            v = norm(v_rel, axis=0)
            
            # Constants for drag coeff
            mach = v / sos # Mach Number
            re = reynolds(mach, rho_a, dyn_vis, 2 * np.sqrt(CA / np.pi)) # Reynolds Number
            kn = knudsen(mach, re) # Knudsen Number
            Cd = dragcoef(re, mach, kn) # Drag Coefficient
            #Cd = 2.0 # Approximation
        
            # Total drag perturbation
            a_drag = -1.0 / (2 * M) * rho_a * Cd * CA * v * v_rel  
            
        else:
            # Total drag perturbation
            a_drag = np.nan

        ''' Gravitational J2 Perturbation '''
        # Gravitational perturbation components
        if r < R_SOI:
            k = 3 * mu_e * J2 * R_e**2 / (2 * r**5)
            J2_x = k * Pos_ECI[0] * (5 * Pos_ECI[2]**2 / r**2 - 1)
            J2_y = k * Pos_ECI[1] * (5 * Pos_ECI[2]**2 / r**2 - 1)
            J2_z = k * Pos_ECI[2] * (5 * Pos_ECI[2]**2 / r**2 - 3)
            a_J2 = np.vstack((J2_x, J2_y, J2_z))
        else:
            a_J2 = np.nan

        ''' Third Body Perturbations: Moon '''
        # Position vector from Earth to Moon (m)
        moon_gcrs = get_body('moon',T)
        rho_m = np.vstack(moon_gcrs.transform_to(GCRS(obstime=T)).cartesian.xyz.to(u.meter).value)

        # Third body perturbation acceleration (Moon) -ECI frame
        # a_moon = -mu_m * (Pos_ECI - rho_m) / (norm(Pos_ECI - rho_m))**3
        a_moon = ThirdBodyPerturbation(Pos_ECI, rho_m, mu_m)

        ''' Total Perturbing Acceleration '''
        if np.isnan(norm(a_drag)) and np.isnan(norm(a_J2)):
            a_tot = a_grav + a_sun + a_moon
        elif np.isnan(norm(a_drag)):
            a_tot = a_grav + a_sun + a_moon + a_J2
        else:
            a_tot = a_grav + a_sun + a_moon + a_J2 + a_drag
            
        # Record perturbtions [time, sun, earth, moon, J2, drag]
        # print([norm(a_moon), mu_m / (norm(Pos_ECI - rho_m))**2]) #<----sort this out
        
        # global Pert
        # SUN = mu_s / (norm(Pos_ECI - rho_s))**2
        # MOON = mu_m / (norm(Pos_ECI - rho_m))**2
        # Pert = np.hstack((Pert, np.vstack((t/(24*60*60), SUN, 
        #     norm(a_grav), MOON, norm(a_J2), norm(a_drag))) ))

    else:
        ''' Total Perturbing Acceleration '''
        a_tot = a_grav + a_sun

    ''' State Rate Equation '''
    X_dot = np.vstack((Vel_ECI, a_tot, 0.0, 0.0))

    return X_dot.flatten()

from scipy.integrate import ode
def Propagate_ECI(t0, X0, Perturbations):

    rmax = norm(X0[:3])  # Initial radius

    state, T_rel = [], []
    def solout(t, X):
        state.extend([X.copy()]); T_rel.extend([t])

        # Determine current radius
        r = norm(X[:3])

        # Print progress
        sys.stdout.write('\r%.3f%%' % (r / (10 * R_SOI) * 100))
        sys.stdout.flush()

        # Record the maximum radius reached
        nonlocal rmax
        if r > rmax: rmax = r

        # Return statement
        if r > R_SOI*10: # Reached one SOI
            return -1 # Ends the integration
        elif r < rmax / 2:
            print('Your meteoroid is geocentric!')
            return -1 # Ends the integration
        else:
            return 0 # Continues integration

    # Setup integrator
    solver = ode(EarthDynamics).set_integrator('dopri5', atol=1e-6)
    solver.set_solout(solout)
    solver.set_initial_value(X0.flatten(), 0).set_f_params([t0, Perturbations])
    
    ''' Numerically integrate until its beyond the Earth's SOI '''
    # t_tcm4 = Time('2010-06-09T06:04:00.0', format='isot', scale='utc').jd # Time of TCM4
    # t_max = (t_tcm4 - t0)*24*60*60
    # solver.integrate(t_max)
    t_max = -np.inf; solver.integrate(t_max)

    # Assign the variables
    t = np.array(T_rel)/(24*60*60) + t0
    X = np.array(state).T

    return t, X

def Propagate_HCI(t, X, t0):

    dt0 = -(t[-1] - t[-2]) * (24*60*60) # Change direction
    T0_rel = (t[-1] - t0) * (24*60*60)

    state, T_rel = [], []
    def solout(t, X):
        state.extend([X.copy()]); T_rel.extend([t])
        
        # Print progress
        sys.stdout.write('\r%.3f%%' % (t / T0_rel * 100))
        sys.stdout.flush()

        return 0 # Continues integration

    # Setup integrator
    solver = ode(SunDynamics).set_integrator('dopri5', first_step=dt0, atol=1e-6)
    solver.set_solout(solout)
    solver.set_initial_value(X[:,-1:].flatten(), T0_rel).set_f_params(t0)
    
    ''' Numerically integrate until its beyond the Earth's SOI '''
    # t_tcm4 = Time('2010-06-09T06:04:00.0', format='isot', scale='utc').jd # Time of TCM4
    # t_max = (t_tcm4 - t0)*24*60*60
    # solver.integrate(t_max)
    solver.integrate(0)

    # Assign the variables
    t_new = np.array(T_rel)/(24*60*60) + t0
    X_new = np.array(state).T

    t = np.hstack((t, t_new[1:]))
    X = np.hstack((X, X_new[:,1:]))
    
    return t, X

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
    
    ''' Calculate the initial meteor's state in ECI coords '''
    X0 = np.vstack((Pos_ECI, Vel_ECI, M, CA))
    OrbitType = 'Heliocentric'  # Initial assumption

    # Integrate with RK4 until outside Earth's sphere of influence
    if verbose:
        print('Rewinding from Bright Flight...')
    [t, X] = Propagate_ECI(t0, X0, Perturbations)

    [Pos_ECI, Vel_ECI] = [X[:3], X[3:6]]; t_soi = [t[-1]]

    if norm(Pos_ECI[:,-1:]) < R_SOI: # Geocentric
        OrbitType = 'Geocentric' 

        # Convert to geocentric orbital elements
        COE = PosVel2OrbitalElements(Pos_ECI, Vel_ECI, 'Earth', 'Classical')
        ra_corr = np.nan; dec_corr = np.nan; v_g = np.nan

    else: # Heliocentric

        # Convert the ECI position and velocity to HCI coordinates
        [Pos_HCI, Vel_HCI] = ECI2HCI(Pos_ECI, Vel_ECI, t)

        # Convert to heliocentric orbital elements
        X = np.vstack((Pos_HCI, Vel_HCI))        
        
        if verbose:
            print('\rNow entering a Sun centred orbit...')
        [t, X] = Propagate_HCI(t, X, t0)

        [Pos_HCI, Vel_HCI] = [X[:3], X[3:6]]
        
        COE = PosVel2OrbitalElements(Pos_HCI, Vel_HCI, 'Sun', 'Classical')
        [Pos_eci, Vel_eci] = HCI2ECI(Pos_HCI[:,-1:], Vel_HCI[:,-1:], t[-1])
        ra_corr = float(np.arctan2(-Vel_eci[1], -Vel_eci[0]))
        dec_corr = float(np.arcsin(-Vel_eci[2] / norm(Vel_eci)))
        v_g = norm(Vel_eci)

        if COE[0, -1] < 0 and COE[1, -1] > 1:
            OrbitType = 'Hyperbolic'

    print('\r' + OrbitType + ' orbit determined')
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
        if Pert.shape[1] > 1:
            # Plot the perturbations over time
            # global Pert
            PlotPerts(Perts)
               
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
        Pert = np.ones((6,1))

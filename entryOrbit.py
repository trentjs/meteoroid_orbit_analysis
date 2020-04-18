#!/usr/bin/env python
"""
Meteoroid Orbit Determination

Inputs: The initial ECEF position (m), ECEF velocity (m/s), and time (jd).
Outputs: The meteoroid's orbital elements about the Sun, or Earth (a,e,i,omega,OMEGA,theta).
         - using J2000 fundamental epoch
"""

__author__ = "Trent Jansen-Sturgeon, Hadrien A.R. Devillepoix"
__copyright__ = "Copyright 2015-2019, Desert Fireball Network"
__license__ = "MIT"
__version__ = "1.0"
__pipeline_task__ = "orbit"

# Core modules
import os
import argparse
from copy import deepcopy
import re
import logging
import yaml
import subprocess
try:
    from urllib.error import URLError
except ImportError:
    from urllib2 import URLError

# Science modules
from random import gauss
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.table import Table, Column, vstack
from astropy.utils.iers import IERS_A, IERS_A_URL, IERS
from astropy.utils.data import download_file
from astropy.coordinates import EarthLocation
try:
    iers_a_file = download_file(IERS_A_URL, cache=True)
    iers_a = IERS_A.open(iers_a_file)
    IERS.iers_table = iers_a
except:
    print('IERS_A_URL is temporarily unavailable')
    pass

# Custom modules
import dfn_utils
from orbital_utilities import OrbitObject, \
    random_compute_orbit_integration_EOE, \
    random_compute_orbit_ceplecha, random_compute_orbit_integration_posvel, \
    compute_cartesian_velocities_from_radiant#, compute_infinity_radiant


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


# column names
a_col_ = 'semi_major_axis'
e_col_ = 'eccentricity'
i_col_ = 'inclination'
w_col_ = 'argument_periapsis'
W_col_ = 'longitude_ascending_node'
Pi_col_ = 'longitude_perihelion'
theta_col_ = 'true_anomaly'
q_col_ = 'perihelion'
Q_col_ = 'aphelion'

ra_inf_radiant_col_ = 'RA_inf'
dec_inf_radiant_col_ = 'Dec_inf'

ra_corr_radiant_col_ = 'RA_g'
dec_corr_radiant_col_ = 'Dec_g'

ecliptic_latitude_col_ = 'ecliptic_latitude'

v_g_col_ = 'V_g'
t_j_col_ = 'T_j'

orb_type_col_ = 'orbit_type'
camFile_col_ = 'cam_file'
camCodename_col_ = 'cam_codenumber'

a_err_col_ = 'err_semi_major_axis'
e_err_col_ = 'err_eccentricity'
i_err_col_ = 'err_inclination'
w_err_col_ = 'err_argument_periapsis'
W_err_col_ = 'err_longitude_ascending_node'
Pi_err_col_ = 'err_longitude_perihelion'
theta_err_col_ = 'err_true_anomaly'
q_err_col_ = 'err_perihelion'
Q_err_col_ = 'err_aphelion'
ra_inf_radiant_err_col_ = 'err_RA_inf'
dec_inf_radiant_err_col_ = 'err_Dec_inf'
ra_corr_radiant_err_col_ = 'err_RA_g'
dec_corr_radiant_err_col_ = 'err_Dec_g'
v_g_err_col_ = 'err_V_g'


Pert = np.vstack((1,1,1,1,1))


class StateVector(object):
    # TODO FIXME use class methods for constructors
    #@classmethod
    #def from_string(cls, date_as_string):
        #day, month, year = map(int, date_as_string.split('-'))
        #date1 = cls(day, month, year)
        #return date1

    def __init__(self, t0,
                 pos_xyz,
                 vel_xyz=None,
                 velocity_inf=None, velocity_err=0.0,
                 ra_ecef_inf=None, dec_ecef_inf=None,
                 ra_ecef_inf_err=0.0, dec_ecef_inf_err=0.0,
                 M=0, CA=0,
                 err_vel_plus=np.vstack((0.0, 0.0, 0.0)), err_vel_minus=np.vstack((0.0, 0.0, 0.0))):

        self.epoch = t0
        self.position = pos_xyz


        # radiant version
        if ra_ecef_inf is not None:
            self.ra_ecef_inf = float(ra_ecef_inf)
            self.dec_ecef_inf = float(dec_ecef_inf)
            self.ra_ecef_inf_err = float(ra_ecef_inf_err)
            self.dec_ecef_inf_err = float(dec_ecef_inf_err)

            self.velocity_inf = float(velocity_inf)
            self.velocity_err = float(velocity_err)

            if any(np.isnan([self.ra_ecef_inf,
                             self.dec_ecef_inf,
                             self.ra_ecef_inf_err,
                             self.dec_ecef_inf_err,
                             self.velocity_inf,
                             self.velocity_err])):
                raise dfn_utils.DataReductionError("Non numeric data input for state vector.")

            # calculate cartesian vector
            self.vel_xyz = compute_cartesian_velocities_from_radiant(self)
            # # compute inertial apparent radiant
            # self.infinity_radiant = compute_infinity_radiant(self)

        # cartesian version
        elif vel_xyz is not None:
            self.vel_xyz = vel_xyz
            self.err_plus_velocity = err_vel_plus
            self.err_minus_velocity = err_vel_minus

            # # compute inertial apparent radiant
            # self.infinity_radiant = compute_infinity_radiant(self)


        else:
            raise TypeError('BLA')
        #self.epoch.delta_ut1_utc = 0

        self.mass = M
        self.cs_area = CA

    def from_ECEF_radiant(cls):
        raise NotImplementedError('')

    def from_ECI_radiant(cls):
        raise NotImplementedError('Initiation of StateVector from ECI radiant not currently supported.')

    def from_cartesian(cls):
        raise NotImplementedError('')


    """
    Randomise the velocities based in a Gaussian distribution
    DEPRECATED
    """
    def randomize_velocity_vector_cartesian(self):
        self.vel_xyz = np.vstack((gauss(self.vel_xyz[j][0], self.err_plus_velocity[j][0]) for j in [0, 1, 2]))
        # # recompute radiant
        # self.infinity_radiant = compute_infinity_radiant(self)

    # Randomise the velocity vector using radiant and speed as parameters space
    def randomize_velocity_vector(self):
        self.velocity_inf = gauss(self.velocity_inf, self.velocity_err)

        self.ra_ecef_inf = gauss(self.ra_ecef_inf, self.ra_ecef_inf_err)
        self.dec_ecef_inf = gauss(self.dec_ecef_inf, self.dec_ecef_inf_err)

        # re-calculate cartesian vector
        self.vel_xyz = compute_cartesian_velocities_from_radiant(self)
        # # re-compute inertial apparent radiant
        # self.infinity_radiant = compute_infinity_radiant(self)

    def computeOrbit(self, orbit_computation_method='integrate_posvel', PlotOrbit=False, ephem=False):
        if type(PlotOrbit) is bool:
            if orbit_computation_method == 'Ceplecha':
                import orbit_integrator_ceplecha as orbit_integrator
            elif orbit_computation_method == 'integrate_EOE':
                import orbit_integrator_EOE as orbit_integrator
            elif orbit_computation_method == 'integrate_posvel':
                import orbit_integrator_posvel as orbit_integrator
            else:
                raise NotImplementedError(orbit_computation_method + ' is not implemented')
            [self.orbit, self.ephem_dict] = orbit_integrator.propagateOrbit(self, 
                                            Plot=PlotOrbit, verbose=False, ephem=ephem)
        else:
            raise NameError(PlotOrbit + ' is not a valid boolean input')
        return self.orbit

    def __str__(self):
        return 'Reference position: ({x:.0f}, {y:.0f}, {z:.0f}) m \nReference time: {isot}Z = {jd:.6f}\nRadiant: ECEF (α, δ) = ({ra_ecef:.5f} ± {ra_ecef_err:.5f}, {dec_ecef:.5f} ± {dec_ecef_err:.5f})\nVelocity: {v_inf:.0f} ± {v_inf_err:.0f}'.format(x=self.position[0][0],y=self.position[1][0],z=self.position[2][0],
                                                        isot=Time(self.epoch, format='jd').isot,jd=self.epoch,
                                                        ra_ecef=self.ra_ecef_inf, ra_ecef_err=self.ra_ecef_inf_err,
                                              dec_ecef=self.dec_ecef_inf, dec_ecef_err=self.dec_ecef_inf_err,
                                              v_inf=self.velocity_inf, v_inf_err=self.velocity_err)
        #return 'vel: {0:.3f} \n ra_ecef: {1:.3f} \n dec_ecef: {2:.3f}'.format(self.velocity_inf, self.ra_ecef_inf, self.dec_ecef_inf)


def read_state_vector_from_dic(dic):
    '''
    Factory function for reading in a state vector from a yaml key parameters sub dictionary
    '''
    logger = logging.getLogger('orbit')


    try:
        earth_loc = EarthLocation(lat=dic['initial_latitude'] * u.deg,
                                lon=dic['initial_longitude'] * u.deg,
                                height=dic['initial_height'] * u.m)

        Pos_geo = np.vstack((earth_loc.x.value,
                            earth_loc.y.value,
                            earth_loc.z.value))

        logger.debug('Reference position: ({lat:.5f}, {lon:.5f}, {hei:.0f}) = ({x:.0f}, {y:.0f}, {z:.0f})'.format(lat=earth_loc.lat,lon=earth_loc.lon,hei=earth_loc.height,
                                                                                                              x=earth_loc.x,y=earth_loc.y,z=earth_loc.z))


        # Time
        try:
            t0 = Time(dic['first_datetime']).jd
        except:
            t0 = Time(dic['datetime']).jd

        logger.debug('Reference time: {isot}Z = {jd:.6f}'.format(isot=Time(t0, format='jd').isot,jd=t0))

        ra_ecef_inf = dic['triangulation_ra_ecef_inf']
        ra_ecef_inf_err = dic['triangulation_ra_ecef_inf_err']
        dec_ecef_inf = dic['triangulation_dec_ecef_inf']
        dec_ecef_inf_err = dic['triangulation_dec_ecef_inf_err']

        logger.debug('Radiant: ECEF (α, δ) = ({ra_ecef:.5f} ± {ra_ecef_err:.5f},\
 {dec_ecef:.5f} ± {dec_ecef_err:.5f})'.format(ra_ecef=ra_ecef_inf, ra_ecef_err=ra_ecef_inf_err*u.deg,
                                              dec_ecef=dec_ecef_inf, dec_ecef_err=dec_ecef_inf_err*u.deg))

        velocity_inf = dic['initial_velocity']
        velocity_err = dic['err_initial_velocity']

        logger.debug('Velocity: {v_inf:.0f} ± {v_inf_err:.0f}'.format(v_inf=velocity_inf*u.m/u.s, v_inf_err=velocity_err*u.m/u.s))

        sv = StateVector(t0, Pos_geo,
                            ra_ecef_inf=ra_ecef_inf, ra_ecef_inf_err=ra_ecef_inf_err,
                            dec_ecef_inf=dec_ecef_inf, dec_ecef_inf_err=dec_ecef_inf_err,
                            velocity_inf=velocity_inf, velocity_err=velocity_err)

    except KeyError as e:
        logger.error('Cannot generate state vector, dictionary is missing some keys')
        raise e


    #print(sv)

    return sv




def read_state_vector_from_triangulated_data(tri_data):
    '''
    Factory function for reading in a state vector from triangulation data

    '''

    #print(tri_data.colnames)

    i = np.argmax(tri_data['height'])
    Pos_geo = np.vstack((tri_data['X_geo'][i],
                         tri_data['Y_geo'][i],
                         tri_data['Z_geo'][i]))


    # Time
    t0 = Time(tri_data['datetime'][i], format='isot', scale='utc').jd

    # least-squares version: uses the position and velocity given by the model
    if tri_data.meta['triangulation_method'] == 'LS' or tri_data.meta['triangulation_method'] == 'UKF':
        sv = StateVector(t0, Pos_geo,
                        ra_ecef_inf=tri_data.meta['triangulation_ra_ecef_inf'], ra_ecef_inf_err=tri_data.meta['triangulation_ra_ecef_inf_err'],
                        dec_ecef_inf=tri_data.meta['triangulation_dec_ecef_inf'], dec_ecef_inf_err=tri_data.meta['triangulation_dec_ecef_inf_err'],
                        velocity_inf=tri_data['D_DT_geo'][i], velocity_err=tri_data['D_DT_err'][i])
    
    # radiant version: needs radiant computed with errors, and EKS run
    elif 'triangulation_ra_ecef_inf' in tri_data.meta and dfn_utils.is_type_pipeline(tri_data, 'velocitic_modeled'):
        sv = StateVector(t0, Pos_geo,
                        ra_ecef_inf=tri_data.meta['triangulation_ra_ecef_inf'], ra_ecef_inf_err=tri_data.meta['triangulation_ra_ecef_inf_err'],
                        dec_ecef_inf=tri_data.meta['triangulation_dec_ecef_inf'], dec_ecef_inf_err=tri_data.meta['triangulation_dec_ecef_inf_err'],
                        velocity_inf=tri_data['D_DT_EKS'][i], velocity_err=tri_data['err_plus_D_DT_EKS'][i])

    # cartesian version
    elif 'DX_DT_geo' in tri_data.colnames:
        Vel_geo = np.vstack((tri_data['DX_DT_geo'][i],
                            tri_data['DY_DT_geo'][i],
                            tri_data['DZ_DT_geo'][i]))
        Err_plus_Vel_geo = np.vstack((tri_data['err_plus_DX_DT'][i],
                                    tri_data['err_plus_DY_DT'][i],
                                    tri_data['err_plus_DZ_DT'][i]))
        Err_minus_Vel_geo = np.vstack((tri_data['err_minus_DX_DT'][i],
                                    tri_data['err_minus_DY_DT'][i],
                                    tri_data['err_minus_DZ_DT'][i]))

        sv = StateVector(t0, Pos_geo,
                        vel_xyz=Vel_geo, err_vel_plus=Err_plus_Vel_geo, err_vel_minus=Err_minus_Vel_geo)
    else:
        raise KeyError('Cannot find entry vector initialisation data')


    return sv


def state_vector_factory(input_object):

    logger = logging.getLogger('orbit')
    logger.debug('This is the state vector factory')

    if isinstance(input_object, str):
        # assume file
        # Extract the triangulated data from file
        logger.debug('Reading {} assuming it is a camera observation file'.format(os.path.basename(input_object)))
        tri_data = Table.read( input_object, format='ascii.ecsv', guess=False, delimiter=',')
        sv = read_state_vector_from_triangulated_data(tri_data)
        telescope = tri_data.meta['telescope']


    elif isinstance(input_object, dict):
        logger.debug('Looking up dictionary for radiant information')
        sv = read_state_vector_from_dic(input_object)
        telescope = input_object['telescope']

    else:
        logger.error('Cannot determine type of input object for generating state vector')
        raise dfn_utils.WrongTableTypeException('Cannot determine type of input object for generating state vector')

    return sv, telescope




def dfn_event_orbit(input_object, KPTable=None, orbitsFile=None, n_MC_orbit=0,
                        orbitmethod='integrate_EOE', PlotOrbit=False, ephem=False):
    '''
    Main function to be called in DFN pipeline
    '''
    logger = logging.getLogger('orbit')

    # determine if MC simulations are needed
    MC = (n_MC_orbit > 0)
    logger.debug('Monte Carlo orbital simulations activated: ' + str(MC))

    if KPTable is None:
        KPTable = Table()



    #FIXME
    #logging.info('Starting orbit computation from {0} triangulated data'.format(tri_data.meta['telescope']))
    # load up the base state vector
    sv, telescope = state_vector_factory(input_object)

    # just keep the number
    camCodename = int(re.sub("[^0-9]", "", telescope))





    if MC:
        # create deep copies of the orginal state vector
        SVList = []
        for i in range(n_MC_orbit):
            SVList += [deepcopy(sv)]

        import multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        logger.info('Spawning {1:d} MC integrations on {0:d} cores'.format(multiprocessing.cpu_count(), n_MC_orbit))

        try:
            if orbitmethod == 'integrate_EOE':
                SVList2 = pool.map(random_compute_orbit_integration_EOE, SVList)
            elif orbitmethod == 'Ceplecha':
                SVList2 = pool.map(random_compute_orbit_ceplecha, SVList)
            elif orbitmethod == 'integrate_posvel':
                SVList2 = pool.map(random_compute_orbit_integration_posvel, SVList)
            else:
                raise NotImplementedError('Orbit method not implemented')
        except (OSError, URLError):
            logger.error("Cant download IERS file")
            raise

        # compute errors
        a_mean = np.mean(u.Quantity([s.orbit.semi_major_axis for s in SVList2]))
        a_std = np.std(u.Quantity([s.orbit.semi_major_axis for s in SVList2]))
        logger.info("Semi-major axis:             {} +- {}".format(a_mean, a_std))
        e_mean = np.mean(u.Quantity([s.orbit.eccentricity for s in SVList2]))
        e_std = np.std(u.Quantity([s.orbit.eccentricity for s in SVList2]))
        logger.info("Eccentricity:                {} +- {}".format(e_mean, e_std))
        i_mean = np.mean(u.Quantity([s.orbit.inclination for s in SVList2]))
        i_std = np.std(u.Quantity([s.orbit.inclination for s in SVList2]))
        logger.info("Inclination:                 {} +- {}".format(i_mean, i_std))
        o_mean = np.mean(u.Quantity([s.orbit.argument_periapsis for s in SVList2]))
        o_std = np.std(u.Quantity([s.orbit.argument_periapsis for s in SVList2]))
        logger.info("Argument of perihelion:      {} +- {}".format(o_mean, o_std))
        O_mean = np.mean(u.Quantity([s.orbit.longitude_ascending_node for s in SVList2]))
        O_std = np.std(u.Quantity([s.orbit.longitude_ascending_node for s in SVList2]))
        logger.info("Longitude of ascending node: {} +- {}".format(O_mean, O_std))
        t_mean = np.mean(u.Quantity([s.orbit.true_anomaly for s in SVList2]))
        t_std = np.std(u.Quantity([s.orbit.true_anomaly for s in SVList2]))
        logger.info("True anomaly:                {} +- {}".format(t_mean, t_std))
        q_mean = np.mean(u.Quantity([s.orbit.perihelion for s in SVList2]))
        q_std = np.std(u.Quantity([s.orbit.perihelion for s in SVList2]))
        logger.info("Perihelion distance:         {} +- {}".format(q_mean, q_std))
        Q_mean = np.mean(u.Quantity([s.orbit.aphelion for s in SVList2]))
        Q_std = np.std(u.Quantity([s.orbit.aphelion for s in SVList2]))
        logger.info("Aphelion distance:           {} +- {}".format(Q_mean, Q_std))
        Pi_mean = np.mean(u.Quantity([s.orbit.longitude_perihelion for s in SVList2]))
        Pi_std = np.std(u.Quantity([s.orbit.longitude_perihelion for s in SVList2]))
        logger.info("Longitude of perihelion:     {} +- {}".format(Pi_mean, Pi_std))

        # inf_radiant_ra_mean = np.mean(u.Quantity([s.infinity_radiant.ra for s in SVList2]))
        # inf_radiant_ra_std = np.std(u.Quantity([s.infinity_radiant.ra for s in SVList2]))
        # logger.info("RA radiant inf:              {} +- {}".format(inf_radiant_ra_mean, inf_radiant_ra_std))
        # inf_radiant_dec_mean = np.mean(u.Quantity([s.infinity_radiant.dec for s in SVList2]))
        # inf_radiant_dec_std = np.std(u.Quantity([s.infinity_radiant.dec for s in SVList2]))
        # logger.info("Dec radiant inf:             {} +- {}".format(inf_radiant_dec_mean, inf_radiant_dec_std))

        corr_radiant_ra_mean = np.mean(u.Quantity([s.orbit.corr_radiant_ra for s in SVList2]))
        corr_radiant_ra_std = np.std(u.Quantity([s.orbit.corr_radiant_ra for s in SVList2]))
        logger.info("RA radiant geo:              {} +- {}".format(corr_radiant_ra_mean, corr_radiant_ra_std))
        corr_radiant_dec_mean = np.mean(u.Quantity([s.orbit.corr_radiant_dec for s in SVList2]))
        corr_radiant_dec_std = np.std(u.Quantity([s.orbit.corr_radiant_dec for s in SVList2]))
        logger.info("Dec radiant geo:             {} +- {}".format(corr_radiant_dec_mean, corr_radiant_dec_std))

        v_g_mean = np.mean(u.Quantity([s.orbit.velocity_g for s in SVList2]))
        v_g_std = np.std(u.Quantity([s.orbit.velocity_g for s in SVList2]))
        logger.info("Geo velocity:                {} +- {}".format(v_g_mean, v_g_std))

    else: # not MC
        SVList2 = [sv]
        try:
            sv.computeOrbit(orbit_computation_method=orbitmethod, 
                PlotOrbit=PlotOrbit, ephem=ephem)
            print("\nCalculated Solution")
            print(sv.orbit)
        except (OSError, URLError):
            # retry once
            try:
                import time
                logger.info('Cannot contact IERS server, re-trying in 5 seconds')
                time.sleep(5)
                sv.computeOrbit(orbit_computation_method=orbitmethod, 
                    PlotOrbit=PlotOrbit, ephem=ephem)
                print("Calculated Solution")
                print(sv.orbit)
            except (OSError, URLError):
                logger.error("Cant download IERS file")
                exit(3)

        if ephem: # Save the ephemeris file
            ephem_table = Table(sv.ephem_dict)
            ephem_file = '.'.join(orbitsFile.split('.')[:-1]) + '_ephemeris.csv'
            print('Saving ephemeris file to {}'.format(ephem_file))
            ephem_table.write(ephem_file, format='ascii.csv', delimiter=',', overwrite=True)




    possible_newcols = [orb_type_col_, camFile_col_, camCodename_col_,
                        a_col_, e_col_, i_col_, w_col_, W_col_, 
                        theta_col_, q_col_, Q_col_, Pi_col_, 
                        # ra_inf_radiant_col_, dec_inf_radiant_col_, 
                        ra_corr_radiant_col_, dec_corr_radiant_col_, 
                        v_g_col_, a_err_col_, e_err_col_,
                        i_err_col_, w_err_col_, W_err_col_, theta_err_col_,
                        q_err_col_, Q_err_col_, Pi_err_col_,
                        # ra_inf_radiant_err_col_, dec_inf_radiant_err_col_,
                        ra_corr_radiant_err_col_, dec_corr_radiant_err_col_,
                        v_g_err_col_, t_j_col_ ]

    for c in possible_newcols:
        if c in KPTable.colnames:
            KPTable.remove_columns(c)

    if MC:
        a_err_col = Column(name=a_err_col_, data=u.Quantity([a_std]))
        e_err_col = Column(name=e_err_col_, data=u.Quantity([e_std]))
        i_err_col = Column(name=i_err_col_, data=u.Quantity([i_std]))
        w_err_col = Column(name=w_err_col_, data=u.Quantity([o_std]))
        W_err_col = Column(name=W_err_col_, data=u.Quantity([O_std]))
        Pi_err_col = Column(name=Pi_err_col_, data=u.Quantity([Pi_std]))
        theta_err_col = Column(name=theta_err_col_, data=u.Quantity([t_std]))
        q_err_col = Column(name=q_err_col_, data=u.Quantity([q_std]))
        Q_err_col = Column(name=Q_err_col_, data=u.Quantity([Q_std]))
        # ra_inf_radiant_err_col = Column(name=ra_inf_radiant_err_col_,
        #                             data=u.Quantity([inf_radiant_ra_std]))
        # dec_inf_radiant_err_col = Column(name=dec_inf_radiant_err_col_,
        #                             data=u.Quantity([inf_radiant_dec_std]))
        ra_corr_radiant_err_col = Column(name=ra_corr_radiant_err_col_,
                                    data=u.Quantity([corr_radiant_ra_std]))
        dec_corr_radiant_err_col = Column(name=dec_corr_radiant_err_col_,
                                    data=u.Quantity([corr_radiant_dec_std]))
        v_g_err_col = Column(name=v_g_err_col_, data=u.Quantity([v_g_std]))

        a_col = Column(name=a_col_, data=u.Quantity([a_mean]))
        e_col = Column(name=e_col_, data=u.Quantity([e_mean]))
        i_col = Column(name=i_col_, data=u.Quantity([i_mean]))
        w_col = Column(name=w_col_, data=u.Quantity([o_mean]))
        W_col = Column(name=W_col_, data=u.Quantity([O_mean]))
        Pi_col = Column(name=Pi_col_, data=u.Quantity([Pi_mean]))
        theta_col = Column(name=theta_col_, data=u.Quantity([t_mean]))
        q_col = Column(name=q_col_, data=u.Quantity([q_mean]))
        Q_col = Column(name=Q_col_, data=u.Quantity([Q_mean]))
        # ra_inf_radiant_col = Column(name=ra_inf_radiant_col_,
        #                             data=u.Quantity([inf_radiant_ra_mean]))
        # dec_inf_radiant_col = Column(name=dec_inf_radiant_col_,
        #                             data=u.Quantity([inf_radiant_dec_mean]))
        ra_corr_radiant_col = Column(name=ra_corr_radiant_col_,
                                    data=u.Quantity([corr_radiant_ra_mean]))
        dec_corr_radiant_col = Column(name=dec_corr_radiant_col_,
                                    data=u.Quantity([corr_radiant_dec_mean]))

        median_orbit = OrbitObject('Heliocentric',a_mean, e_mean,i_mean, o_mean,
                                    O_mean, t_mean, ra_corr=corr_radiant_ra_mean,
                                    dec_corr=corr_radiant_dec_mean)

        ecliptic_latitude_col = Column(name=ecliptic_latitude_col_,
                                    data=u.Quantity([median_orbit.ecliptic_latitude]))
        v_g_col = Column(name=v_g_col_, data=u.Quantity([v_g_mean]))
        t_j_col = Column(name=t_j_col_, data=u.Quantity([median_orbit.T_j]))

        # if all found orbits are of the same type: all good. if different type, mark as Unstable
        if all(SVList2[0].orbit.orbit_type == s.orbit.orbit_type for s in SVList2):
            orbtype = SVList2[0].orbit.orbit_type
        else:
            orbtype = 'Unstable'
            median_orbit.orbit_type = 'Unstable'
        orb_type_col = Column(name=orb_type_col_, data=[orbtype])

    else: #not MC

        a_col = Column(name=a_col_, data=u.Quantity([sv.orbit.semi_major_axis]))
        e_col = Column(name=e_col_, data=u.Quantity([sv.orbit.eccentricity]))
        i_col = Column(name=i_col_, data=u.Quantity([sv.orbit.inclination]))
        w_col = Column(name=w_col_, data=u.Quantity([sv.orbit.argument_periapsis]))
        W_col = Column(name=W_col_, data=u.Quantity([sv.orbit.longitude_ascending_node]))
        Pi_col = Column(name=Pi_col_, data=u.Quantity([sv.orbit.longitude_perihelion]))
        theta_col = Column(name=theta_col_, data=u.Quantity([sv.orbit.true_anomaly]))
        q_col = Column(name=q_col_, data=u.Quantity([sv.orbit.perihelion]))
        Q_col = Column(name=Q_col_, data=u.Quantity([sv.orbit.aphelion]))
        orb_type_col = Column(name=orb_type_col_, data=[sv.orbit.orbit_type])
        # ra_inf_radiant_col = Column(name=ra_inf_radiant_col_,
        #                             data=u.Quantity(sv.infinity_radiant.ra))
        # dec_inf_radiant_col = Column(name=dec_inf_radiant_col_,
        #                             data=u.Quantity(sv.infinity_radiant.dec))
        ra_corr_radiant_col = Column(name=ra_corr_radiant_col_,
                                    data=u.Quantity([sv.orbit.corr_radiant_ra]))
        dec_corr_radiant_col = Column(name=dec_corr_radiant_col_,
                                    data=u.Quantity([sv.orbit.corr_radiant_dec]))
        ecliptic_latitude_col = Column(name=ecliptic_latitude_col_,
                                    data=u.Quantity([sv.orbit.ecliptic_latitude]))
        v_g_col = Column(name=v_g_col_, data=u.Quantity([sv.orbit.velocity_g]))
        t_j_col = Column(name=t_j_col_, data=u.Quantity([sv.orbit.T_j]))

    KPTable.add_columns( [orb_type_col, a_col, e_col, i_col, w_col, W_col,
                          theta_col, q_col, Q_col, Pi_col,
                          # ra_inf_radiant_col, dec_inf_radiant_col,
                          ra_corr_radiant_col, dec_corr_radiant_col,
                          v_g_col,
                          t_j_col])

    if MC:
        KPTable.add_columns([a_err_col, e_err_col, i_err_col, w_err_col, W_err_col,
                             theta_err_col, q_err_col, Q_err_col, Pi_err_col,
                             # ra_inf_radiant_err_col, dec_inf_radiant_err_col,
                             ra_corr_radiant_err_col, dec_corr_radiant_err_col,
                             v_g_err_col])


    # define new orbits columns
    a_col = Column(name=a_col_, data=u.Quantity([s.orbit.semi_major_axis for s in SVList2]))
    e_col = Column(name=e_col_, data=u.Quantity([s.orbit.eccentricity for s in SVList2]))
    i_col = Column(name=i_col_, data=u.Quantity([s.orbit.inclination for s in SVList2]))
    w_col = Column(name=w_col_, data=u.Quantity([s.orbit.argument_periapsis for s in SVList2]))
    W_col = Column(name=W_col_, data=u.Quantity([s.orbit.longitude_ascending_node for s in SVList2]))
    theta_col = Column(name=theta_col_, data=u.Quantity([s.orbit.true_anomaly for s in SVList2]))
    #q_col = Column(name=q_col_, data=u.Quantity([s.orbit.perihelion for s in SVList2]))
    #Q_col = Column(name=Q_col_, data=u.Quantity([s.orbit.aphelion for s in SVList2]))
    orb_type_col = Column(name=orb_type_col_, data=[s.orbit.orbit_type for s in SVList2])
    camCodename_col = Column(name=camCodename_col_,
                             data=[camCodename for k in range(len(SVList2))],
                             description='Camera code number used to compute this orbit')

    orbTable = Table([orb_type_col, a_col, e_col, i_col, w_col, W_col, theta_col,
                      #                  q_col,Q_col,
                      camCodename_col])

    # read possible previous orbits file
    if orbitsFile and os.path.isfile(orbitsFile):
        logger.info('Reading previous orbit(s) file: ' + orbitsFile)
        oldOrbTable = Table.read(orbitsFile, format='ascii.csv', guess=False, delimiter=',')
        # merge with the new orbits
        logger.info('Appending new orbital solutions')
        orbTable = vstack([oldOrbTable, orbTable])
    if orbitsFile:
        # write all the orbits
        logger.info('Writing calculated orbit(s) to file: ' + orbitsFile)
        orbTable.write(orbitsFile, format='ascii.csv', delimiter=',',
                        fill_values=[('nan', '')], fast_writer=False, overwrite=True)

    if MC:
        orbital_parameters = {'semi_major_axis': a_mean,
                             'eccentricity': e_mean,
                             'inclination': i_mean,
                             'argument_periapsis': o_mean,
                             'longitude_ascending_node': O_mean,
                             'longitude_perihelion': Pi_mean,
                             'true_anomaly': t_mean,
                             'perihelion': q_mean,
                             'aphelion': Q_mean,

                             # 'RA_inf': inf_radiant_ra_mean,
                             # 'Dec_inf': inf_radiant_dec_mean,

                             'RA_g': corr_radiant_ra_mean,
                             'Dec_g': corr_radiant_dec_mean,

                             'V_g': v_g_mean,
                             'T_j': median_orbit.T_j,

                             'ecliptic_latitude': median_orbit.ecliptic_latitude,

                             'orbit_type': median_orbit.orbit_type,

                             'err_semi_major_axis': a_std,
                             'err_eccentricity': e_std,
                             'err_inclination': i_std,
                             'err_argument_periapsis': o_std,
                             'err_longitude_ascending_node': O_std,
                             'err_longitude_perihelion': Pi_std,
                             'err_true_anomaly': t_std,
                             'err_perihelion': q_std,
                             'err_aphelion': Q_std,
                             # 'err_RA_inf': inf_radiant_ra_std,
                             # 'err_Dec_inf': inf_radiant_dec_std,
                             'err_RA_g': corr_radiant_ra_std,
                             'err_Dec_g': corr_radiant_dec_std,
                             'err_V_g': v_g_std }
    else: #not MC
         orbital_parameters = {'semi_major_axis': sv.orbit.semi_major_axis,
                             'eccentricity': sv.orbit.eccentricity,
                             'inclination': sv.orbit.inclination,
                             'argument_periapsis': sv.orbit.argument_periapsis,
                             'longitude_ascending_node': sv.orbit.longitude_ascending_node,
                             'longitude_perihelion': sv.orbit.longitude_perihelion,
                             'true_anomaly': sv.orbit.true_anomaly,
                             'perihelion': sv.orbit.perihelion,
                             'aphelion': sv.orbit.aphelion,

                             # 'RA_inf': sv.infinity_radiant.ra,
                             # 'Dec_inf': sv.infinity_radiant.dec,

                             'RA_g': sv.orbit.corr_radiant_ra,
                             'Dec_g': sv.orbit.corr_radiant_dec,

                             'V_g': sv.orbit.velocity_g,
                             'T_j': sv.orbit.T_j,

                             'ecliptic_latitude': sv.orbit.ecliptic_latitude,

                             'orbit_type': sv.orbit.orbit_type}

    orbital_parameters['processing_orbit_version'] = float(__version__)

    # plot orbit
    if orbitsFile:
        try:
            scp_dir = os.path.dirname(os.path.realpath(__file__))
            subprocess.check_output([os.path.join(scp_dir, "orbit_gnuplot_KP.sh"), "-x", "0", "-i", orbitsFile])
            logger.info('Successfully generated orbital plot')
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error('Could not generate orbit plot')



    return KPTable, orbTable, orbital_parameters

def main(triang_folder, overwrite=True):
    log_file = dfn_utils.create_logger(triang_folder, 'orbit')
    logger = logging.getLogger('orbit')
    print('Started logging to: ' + log_file)

    

    # look for KP yaml
    dir_list = os.listdir(triang_folder)
    KPs = sorted([d for d in dir_list if '_key_parameters.yaml' in d])

    if len(KPs) < 1:
        raise FileNotFoundError('Cannot find a yaml file in triangulation folder')

    yaml_file = os.path.join(triang_folder, KPs[-1])

    logger.info('Using key parameters file: {}'.format(yaml_file))

    # read KP yaml
    kp = yaml.safe_load(open(yaml_file, 'r'))

    # check that an orbit has not already been calculated
    if not overwrite and 'semi_major_axis' in kp['all']:
        raise ValueError('Orbit already calculated and overwrite set to FALSE')

    # generate all orbits out file name
    EC_found, event_codename = dfn_utils.event_codename_matcher(inputdirectory)
    if not EC_found:
        logger.error('Cannot determine event codename')
        event_codename = 'fireball'
    ofile = os.path.join(triang_folder, event_codename + "_MC_orbits.csv")
    logger.info('Appending individual orbital solutions to: {}'.format(ofile))

    # scroll though each camera
    for k in kp:
        if  (k != 'all') and not allcams:
            continue

        # for all consolidated entry data, create dummy observer
        if k == 'all':
            kp[k]['telescope'] = '00_all'

        try:
            logger.info('Calculating orbit using data from {}'.format(kp[k]['telescope']))

            KPTable, orbTable, orb_para = dfn_event_orbit(kp[k], KPTable=None,
                                        orbitsFile=ofile, n_MC_orbit=n_MC_orbit,
                                        orbitmethod=args.propagationmethod,
                                        PlotOrbit=args.PlotOrbit, ephem=args.save_ephemeris)

            dfn_utils.sanitize_dictionary_for_ascii_write(orb_para)
            kp[k].update(orb_para)


        except dfn_utils.DataReductionError as e:
            logger.error(e)
            logger.error('Missing information to compute orbit with {} data'.format(k))
            continue

    with open(yaml_file, 'w') as outfile:
        yaml.dump(kp, outfile)
    logger.info('Orbit analysis main results added to: {}'.format(yaml_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate meteor orbits')
    inputgroup = parser.add_mutually_exclusive_group(required=True)
    inputgroup.add_argument("-d", "--inputdirectory", type=str,
            help="input directory of triangulated folder")
    inputgroup.add_argument("-i", "--inputfile", type=str,
                            help="input file (ecsv or yaml)")
    parser.add_argument("-n", "--MCorbit", type=int, default=0,
                            help="number of Monte Carlo simulations for the orbit")
    parser.add_argument("-O", "--propagationmethod", type=str, help="Propagation method",
            choices=['Ceplecha','integrate_EOE','integrate_posvel'], default='integrate_posvel')
    parser.add_argument("-p", "--PlotOrbit", action="store_true",
                            help="Plot the orbit", default=False)
    parser.add_argument("-a", "--automaticroutine", action="store_true", default=False,
            help="Use this mode to only recalculate a trajectory of possible input files have been changed. This only has an effect if --inputdirectory mode is selected.")
    parser.add_argument("-A", "--allcams", action="store_true", default=False,
            help="Run orbital calculations for all cameras instead of consolidated data")
    parser.add_argument("-ephem", "--save_ephemeris", action="store_true", default=False,
            help="Save a CSV file containing the ephemeris information for the meteoroid in the days leading to impact")
    args = parser.parse_args()

    n_MC_orbit = args.MCorbit
    allcams = args.allcams
    inputdirectory = args.inputdirectory
    automaticroutine = args.automaticroutine

    if inputdirectory and os.path.isdir(inputdirectory):

        if automaticroutine:
            dir_list = os.listdir(inputdirectory)
            traj_folds = sorted([d for d in dir_list if d.startswith('trajectory_auto_20')])
            if len(traj_folds) < 1:
                exit(2)
            triang_folder = os.path.join(inputdirectory, traj_folds[-1])
        else:
            triang_folder = inputdirectory
        
        try:
            main(triang_folder, overwrite=not automaticroutine)
        except FileNotFoundError as e:
            print(e)
            exit(1)
        except ValueError as e:
            print(e)
            exit(5)


    elif args.inputfile and os.path.isfile(args.inputfile):
        # single file
        tri_file = args.inputfile

        log_file = dfn_utils.create_logger(os.path.dirname(tri_file), 'orbit')
        logger = logging.getLogger('orbit')
        print('Started logging to: ' + log_file)

        ofile = os.path.join(os.path.dirname(tri_file),
                        os.path.splitext(tri_file)[0]+'_'+args.propagationmethod+'.csv')

        if os.path.isfile(ofile):
            os.remove(ofile)

        KPTable, orbTable, orb_para = dfn_event_orbit(tri_file, KPTable=None,
                                        orbitsFile=ofile, n_MC_orbit=n_MC_orbit,
                                        orbitmethod=args.propagationmethod,
                                        PlotOrbit=args.PlotOrbit, ephem=args.save_ephemeris)
        print('Output has been appended to: ' + ofile)

    else:
        exit(1)

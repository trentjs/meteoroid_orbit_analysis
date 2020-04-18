"""
Meteoroid Orbital Regression Propagator

Created on Wed Jul 5 18:01:00 2017
@author: Trent Jansen-Sturgeon
"""

# System modules
import os
import sys
import argparse
import time as time_module
import matplotlib.pyplot as plt

# Science modules
import rebound
import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.table import Table, vstack, Column

# Custom modules
from entryOrbit import read_state_vector_from_triangulated_data, AU
from trajectory_utilities import ECEF2ECI, HCRS2HCI

# MPI modules
from mpi4py import MPI


def SolarSystem(t_jd):

    t_iso = Time(t_jd, format='jd', scale='utc').iso
    date = ':'.join(t_iso.split(':')[:-1])

    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.add("Sun", date=date)
    sim.add("Mercury", date=date)
    sim.add("Venus", date=date)
    sim.add("399", date=date) # Earth
    sim.add("301", date=date) # Moon
    sim.add("Mars", date=date)
    sim.add("Jupiter", date=date)
    sim.add("Saturn", date=date)
    sim.add("Uranus", date=date)
    sim.add("Neptune", date=date)
    sim.particles[0].hash = 'sun'
    sim.particles[3].hash = 'earth'
    # sim.convert_particle_units('AU', 'yr', 'Msun')
    sim.move_to_com()
    return sim

if __name__ == '__main__':

    # Identify the processor
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Start the timer for reference
    start = time_module.time()

    # Gather some user defined information
    if rank == 0:
        parser = argparse.ArgumentParser(description='Run a Solar System Regression with N particles.')
        parser.add_argument("-e", "--eventFile", type=str,
                help="Event file for propagation [ECSV]", required=True)
        parser.add_argument("-n","--n_particles",type=int,
                help="Number of particles to run. Must be an integer.", default=1000)
        parser.add_argument("-yr","--n_years",type=float,
                help="Number of years to integrate over.", default=1e6) # 1 Myrs
        parser.add_argument("-o","--n_outputs",type=int,
                help="Number of outputs from the integration.", default=200)
        # parser.add_argument("-c","--comment",type=str,
        #         help="add a version name to appear in saved file titles. If unspecified, _testing_ is used.",default='testing')
        parser.add_argument("-f","--fits",action="store_false",
                help="Generate FITS results file? (Default=True)",default=True)
        parser.add_argument("-m","--mp4",action="store_false",
                help="Generate MP4 animation file? (Default=True)",default=True)
        parser.add_argument("-p","--plot",action="store_false",
                help="Save the plots? (Default=True)",default=True)
        args = parser.parse_args()

        # Assign the inputs
        N_years = args.n_years
        N_outputs = args.n_outputs
        if N_outputs < 2:
            N_outputs = 2
            print('Number of outputs is set to the minimum (2).')
        file_out = args.fits
        movie_out = args.mp4
        Plot = args.plot
        N = args.n_particles

        # Collect all the particle parameters
        N_particles = np.array([N//size]*size)
        N_particles[:(N%size)] += 1

        TriFile = args.eventFile
        TriData = Table.read( TriFile, format='ascii.ecsv', guess=False, delimiter=',')
        sv = read_state_vector_from_triangulated_data(TriData)
        t0 = sv.epoch;              pos_xyz = sv.position
        vel_geo = sv.velocity_inf;  vel_err = sv.velocity_err
        ra_geo  = sv.ra_ecef_inf;   ra_err  = sv.ra_ecef_inf_err
        dec_geo = sv.dec_ecef_inf;  dec_err = sv.dec_ecef_inf_err

        # Parent body state [t,x,y,z,v,ra,dec,v_err,ra_err,dec_err]
        parent = [t0, pos_xyz, vel_geo, ra_geo, dec_geo, vel_err, ra_err, dec_err]

        # Check if the solar system is already simulated
        ss_file = os.path.join(os.path.dirname(TriFile), "solar_system.bin")
        if not os.path.isfile(ss_file): # Setup the Solar System
            sim = SolarSystem(t0) # Resets everything
            sim.save(ss_file) # Creates everything and saves

        for i in range(1, size):
            comm.send([parent, N_particles, N_years, N_outputs, ss_file, file_out, movie_out, Plot], dest=i)

    else:
        [parent, N_particles, N_years, N_outputs, ss_file, file_out, movie_out, Plot] = comm.recv(source=0)

    if not N_particles[rank]: # if there are no particles on core
        print('Terminating additional core.'); exit()

    # Generates the same Solar System on all ranks
    sim = rebound.Simulation(ss_file) # Resets the solar system
    sim.N_active = sim.N

    # Gather Earth parameters
    Pos_earth = np.vstack((sim.particles['earth'].xyz))
    Vel_earth = np.vstack((sim.particles['earth'].vxyz))

    ''' Generate particles '''
    [t0, pos_xyz, vel_geo, ra_geo, dec_geo, vel_err, ra_err, dec_err] = parent
    Vel_geo = np.random.normal(vel_geo, vel_err, size=N_particles[rank])
    Ra_geo = np.deg2rad(np.random.normal(ra_geo, ra_err, size=N_particles[rank]))
    Dec_geo = np.deg2rad(np.random.normal(dec_geo, dec_err, size=N_particles[rank]))

    Vel_xyz = -np.vstack((Vel_geo * np.cos(Ra_geo) * np.cos(Dec_geo),
                          Vel_geo * np.sin(Ra_geo) * np.cos(Dec_geo),
                          Vel_geo * np.sin(Dec_geo)))

    [pos_ECI, Vel_ECI] = ECEF2ECI(pos_xyz, Vel_xyz, t0)
    pos_BCI = Pos_earth + HCRS2HCI(pos_ECI) / AU
    Vel_BCI = Vel_earth + HCRS2HCI(Vel_ECI) / AU*(365.2422*24*60*60)

    for j in range(N_particles[rank]):
        sim.add(x=pos_BCI[0,0], y=pos_BCI[1,0], z=pos_BCI[2,0],
            vx=Vel_BCI[0,j], vy=Vel_BCI[1,j], vz=Vel_BCI[2,j])

    # Integrator options
    sim.integrator = "ias15"
    
    # sim.integrator = "mercurius"
    # sim.integrator = "hermes"
    # sim.testparticle_type = 1

    # sim.integrator = "whfast"#; sim.dt = 0.000005
    # sim.ri_whfast.safe_mode = 0
    # sim.ri_whfast.corrector =  11

    # Collision options
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.collision_resolve_keep_sorted = 1

    ''' Integrate the simulation back in time '''
    times = np.linspace(0,-N_years,N_outputs)
    sma = np.zeros((N_particles[rank], N_outputs))
    ecc = np.zeros((N_particles[rank], N_outputs))
    inc = np.zeros((N_particles[rank], N_outputs))

    for i,time in enumerate(times):
        sim.integrate(time)
        sim.integrator_synchronize()

        # Print progress
        if rank == 0:
            sys.stdout.write('\r%.2f%% Complete' % ((i+1)/N_outputs*100))
            sys.stdout.flush()

        for p in range(-N_particles[rank],0):
            particle = sim.particles[p]
            sma[p,i] = particle.a
            ecc[p,i] = particle.e
            inc[p,i] = particle.inc

    # Get all the N_particles data to master
    data = np.dstack((sma,ecc,inc))
    DATA = np.zeros((np.sum(N_particles), N_outputs, 3))
    sendcounts = np.shape(data)[1]*np.shape(data)[2]*N_particles;
    displacements = np.append(0,np.cumsum(sendcounts)[:-1])
    comm.Gatherv(data, [DATA, sendcounts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:

        # Name the AstroFolder
        import datetime; date_str = datetime.datetime.now().strftime('%Y%m%d')
        AstroFolder = os.path.join(os.path.dirname(TriFile), 'regression_'+date_str)
        if not os.path.isdir(AstroFolder): # Create the directory if it doesn't exist
            os.mkdir(AstroFolder)

        # Name the AstroFile
        print('\n'); i = 1
        AstroFolder = os.path.join(AstroFolder, str(np.sum(N_particles))+
            '_asteroids_integrated_for_'+str(int(N_years))+'yrs_0')
        while os.path.isdir(AstroFolder): # Make sure the folder name is unique
            AstroFolder = '_'.join(AstroFolder.split('_')[:-1])+'_'+str(i); i += 1
        os.mkdir(AstroFolder)

    # All the plots, unless requested not to.
    if rank == 0 and Plot:

        # a_condition = DATA[:,:,0]>0 + DATA[:,:,0]<10
        # e_condition = DATA[:,:,1]>0 + DATA[:,:,1]<1
        SMA = DATA[:,:,0]#[a_condition + e_condition]
        ECC = DATA[:,:,1]#[a_condition and e_condition]
        INC = DATA[:,:,2]#[a_condition and e_condition]

        # Mean Motion Resonances
        MMR = [ 5.20336301 / (3/1)**(2./3), # 3:1 resonance (2.5Myr hl)
                # 5.20336301 / (8/3)**(2./3), # 8:3 resonance (34.0Myr hl)
                5.20336301 / (5/2)**(2./3), # 5:2 resonance (0.6Myr hl)
                # 5.20336301 / (7/3)**(2./3), # 7:3 resonance (19.0Myr hl)
                # 5.20336301 / (2/1)**(2./3), # 2:1 resonance (>100Myr hl)
                ]

        # Planetary Crossing
        a_pc = np.linspace(0.05,10,500)
        e_pc = lambda a_planet, a: abs(a_planet/a - 1)
        E_PC = [e_pc(0.723, a_pc), # Venus crossing orbit
                e_pc(1.000, a_pc), # Earth crossing orbit
                e_pc(1.524, a_pc), # Mars crossing orbit
                e_pc(5.203, a_pc), # Jupiter crossing orbit
                ]

        # Nu6 Resonance (~1Myr hl)
        a_v6 = [2.06, 2.08, 2.115, 2.16, 2.24, 2.315]
        i_v6 = [2.5, 5, 7.5, 10, 12.5, 15]

        # Plot the asteroid distribution comparison
        plt.figure(figsize=(16,9))
        plt.subplot(3,1,1)
        plt.xlabel("Semi-major Axis [AU]")
        plt.ylabel("Number of Particles\nAfter {0:.0f} yrs".format(N_years))
        plt.hist(SMA[:,-1], bins=500, range=(0,5))
        [plt.axvline(x=mmr, c='r') for mmr in MMR]
        plt.axvline(x=np.median(a_v6), c='r')

        plt.subplot(3,1,2)
        plt.xlabel("Eccentricity")
        plt.ylabel("Number of Particles\nAfter {0:.0f} yrs".format(N_years))
        plt.hist(ECC[:,-1], bins=500, range=(0,1))

        plt.subplot(3,1,3)
        plt.xlabel("Inclination [deg]")
        plt.ylabel("Number of Particles\nAfter {0:.0f} yrs".format(N_years))
        plt.hist(np.rad2deg(INC[:,-1]), bins=500, range=(0,180))

        plt.savefig(os.path.join(AstroFolder,'aei_histogram.png'), format='png')

        # Plot the asteroid distribution after
        cond = (SMA[:,-1]>0) & (SMA[:,-1]<10)
        SMA_f = SMA[:,-1][cond]
        ECC_f = ECC[:,-1][cond]
        INC_f = INC[:,-1][cond]

        plt.figure(figsize=(16,9))
        plt.subplot(1,2,1); plt.title('AFTER @ t = -'+str(N_years)+'yrs')
        plt.xlabel("Semi-major Axis [AU]"); plt.gca().set_xlim([0,5])
        plt.ylabel("Eccentricity"); plt.gca().set_ylim([0,1])
        plt.hexbin(np.append(SMA_f,[-0.01,10.01]), np.append(ECC_f,[-0.01,1.01]), gridsize=200, mincnt=1)
        [plt.axvline(x=mmr, c='r') for mmr in MMR]
        [plt.plot(a_pc, e, 'r') for e in E_PC]
        plt.axvline(x=np.median(a_v6), c='r')

        plt.subplot(1,2,2)
        plt.xlabel("Semi-major Axis [AU]"); plt.gca().set_xlim([0,5])
        plt.ylabel("Inclination [deg]"); plt.gca().set_ylim([0,180])
        plt.hexbin(np.append(SMA_f,[-0.01,10.01]), np.append(np.rad2deg(INC_f),[-1,181]), gridsize=200, mincnt=1)
        [plt.axvline(x=mmr, c='r') for mmr in MMR]
        plt.plot(a_v6, i_v6, 'r')

        plt.savefig(os.path.join(AstroFolder,'aei_plot.png'), format='png')

    # Generate FITS file, unless requested not to.
    if rank == 0 and file_out:

        # DATA = np.zeros((N_tot, N_outputs, 3))
        N_tot = np.sum(N_particles)
        ast_col = Column(name='asteroid', data=np.array([np.arange(N_tot)+1]*N_outputs).reshape(N_outputs*N_tot))
        time_col = Column(name='years', data=np.array([times]*N_tot).reshape(N_outputs*N_tot, order='F'))
        DATA_rearranged = DATA.reshape((N_tot*N_outputs,3), order='F')
        a_col = Column(name='semimajor_axis',  data=DATA_rearranged[:,0], unit=u.AU)
        e_col = Column(name='eccentricity', data=DATA_rearranged[:,1])
        i_col = Column(name='inclination', data=np.rad2deg(DATA_rearranged[:,2]), unit=u.deg)

        TriTable = Table([ast_col, time_col, a_col, e_col, i_col])

        AstroFile = os.path.join(AstroFolder,'aei_history.fits')
        TriTable.write(AstroFile, format='fits')

        print('\nAsteroid integration file written to: ' + AstroFile)

    # Make movie, unless requested not to.
    if rank == 0 and movie_out:

        import matplotlib.animation as manimation
        writer = manimation.FFMpegWriter(fps=10, metadata={})

        fig = plt.figure(figsize=(16,9))
        [plt.axvline(x=mmr, c='r') for mmr in MMR]
        [plt.plot(a_pc, e, 'r') for e in E_PC]
        l, = plt.plot([], [], '.b')#, markersize=2)
        t = plt.gca().text(3.5,0.95,'')

        # plt.xlim(1.5, 4); plt.ylim(0, 1)
        plt.xlim(0,10); plt.ylim(0,1)
        plt.xlabel('Semimajor Axis')
        plt.ylabel('Eccentricity')

        # plt.xlim(1.5, 4); plt.ylim(0, 180)
        # plt.xlabel('Semimajor Axis')
        # plt.ylabel('Inclination')

        VideoFile = os.path.join(AstroFolder,'ae_movie.mp4')
        with writer.saving(fig, VideoFile, 200):
            for i in range(N_outputs):
                x = DATA[:,i,0]
                y = DATA[:,i,1]
                l.set_data(x, y)
                t.set_text(str(int(times[i]))+' yrs')
                writer.grab_frame()

                sys.stdout.write('\rAnimating results: %.2f%% Complete' % ((i+1)/N_outputs*100))
                sys.stdout.flush()

        print('\nAsteroid integration animation written to: ' + VideoFile)


    stop = time_module.time()
    print('Core', rank, 'of', size,'finished in',
            np.round(stop - start, 2), 'seconds!')



# for i in *.fits; do stilts plot2plane layer_1=mark in=${i} x_1="semimajor_axis" y_1="eccentricity" ymin=0 ymax=1 xmin=1.5 xmax=4 out="${i%.*}.png"; done
# ffmpeg -r 10 -pattern_type glob -i "*.png" -s 500x400  -vcodec libx264 out.mp4

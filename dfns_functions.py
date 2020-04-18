#@+leo-ver=5-thin
#@+node:martin.20160618083650.153: * @file ~/dfn/server/data_processing/src/triangulation/dfns_functions.py
#@@language python

#@+<<docs>>
#@+node:martin.20130919104130.2231: ** <<docs>>
#python 3 only
#@-<<docs>>
#@+<<copy>>
#@+node:martin.20150320102606.3: ** <<copy>>
# Copyright 2017 Curtin University of Technology, Perth, Western Australia

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#@-<<copy>>

#@+others
#@+node:martin.20130919104130.2230: ** imports
import datetime
import calendar
import socket
import os
import logging
import configparser
import glob
import math
import time
import csv

from astropy.table import Table

logger = logging.getLogger()
#@+node:martin.20130509100437.1849: ** name and today
def name():
    """return the reduced hostname, not full qualified"""
    return socket.gethostname().upper().split('.')[0]
    
def today():
    return str(datetime.datetime.now()).split(' ')[0].strip()

def log_name():
    return today() + '_' + name() + r'_log_'
    
def name_and_today():
    return log_name(), name(), today()
    
def name_from_path( local_path):
    """given a data path, eg
    /data0/DFNSMALL00/2015/08/2015-08-15_DFNSMALL00_12345678/
    separate out system name.
    returns DFNUNKNOWN if nothing found"""
    result = local_path.rstrip( os.sep).split( os.sep)
    result = result[-1].split('_')
    return next((a for a in result if 'DFN' in a), 'DFNUNKNOWN')

def type_from_name( sysname = ''):
    if sysname == '':
        sysname = name()
    if 'ASTRO' in sysname:
        img_str = 'A'
    elif 'KIT' in sysname:
        img_str = 'K' #kit
    elif 'PLUS' in sysname:
        img_str = 'P' #new plus system
    elif 'NEXT' in sysname:
        img_str = 'N' #new plus system
    else:
        img_str = 'S' #small system
    return img_str

def hostname_from_filename( fname):
    #06_2013-11-03_161628_DSC_9931.thumb.masked.altazi.txt
    h = fname.split('_')[0]
    if '_K_' in fname:
        h = 'DFNKIT' + h
    elif '_N_' in fname:
        h = 'DFNEXT' + h
    elif '_P_' in fname:
        h = 'DFNPLUS' + h
    elif '_A_' in fname:
        h = 'DFNASTRO' + h
    else:
        h = 'DFNSMALL' + h
    return h

#@+node:martin.20130712114226.1868: ** event file handling
#@+node:martin.20150918192336.1: *3* loader
def read_event_file( fin):
    """read in an event file to (pts,namelist), strips out extra comments"""
    if fin.lower().endswith('txt'): #old style txt format
        return read_event_txt( fin)
    else: #newer ecsv format
        return read_event_ecsv( fin)
    
#@+node:martin.20150918205803.1: *4* txt file
#station_name,124.567,-31.034,208.2
#[ (datetime,x,y,(-dx,+dx),(-dy,+dy),brightness,(-b,+b) ), (), ...]
#[ (datetime,x,y,(-dx,+dx),(-dy,+dy),brightness,(-b,+b), jd ), (), ...]
def read_event_txt( fin):

    logger = logging.getLogger()
    namelist = ['station','123.0','-31.0','200.0'] #dummy values
    pts = []
    fh = open(fin,'rt')
    for line in fh.readlines():
        if line.startswith('#'):
            data = line.rstrip('\n\r, ').split(',')
            if len(data) == 4: #namestr, may have ',,,,' at the end
                namelist = data[:4] # name still has starting '#'
            else: #some other comment, maybe headstring, ignore it
                pass
        elif '***' in line:
            #some printf overflow in IDL, ignore line
            logger.warning('line read problem, ' + str(line) + ', ' + str(fin) )
            continue
        else:
            data = line.rstrip('\n\r ').split(',')
            if len(data) >= 10: # data line
#            '#datetime,x,y,-dx,+dx,-dy,+dy,brightness,-b,+b\n'
                datetime = data[0]
                try:
                    x = float( data[1])
                    y = float( data[2])
                except ValueError:
                    x = data[1]
                    y = data[2]
                dx = ( float(data[3]), float(data[4]) )
                dy = ( float(data[5]), float(data[6]) )
                brightness = float(data[7])
                db = ( float(data[8]), float(data[9]) )
                line_list = [datetime, x, y, dx, dy, 
                            brightness, db]
                if len(data) == 11: #julian day added as extra column in newer formats
                    line_list.append( float(data[10]))
                pts.append( line_list )
    return pts, namelist
#@+node:martin.20150918205810.1: *4* ecsv file
def read_event_ecsv( fin):
    """read in an ecsv format event file, return pts and namelist"""
    namelist = ['station','123.0','-31.0','200.0'] #dummy values
    pts = []
    #read whole file to get station details
    with open(fin,'rt') as fh:
        for line in fh.readlines():
            if line.startswith('#'):
                if 'camera_codename' in line:
                    namelist[0] = parse_a_line( line)
                if 'obs_longitude' in line:
                    namelist[1] = parse_a_line( line)
                if 'obs_latitude' in line:
                    namelist[2] = parse_a_line( line)
                if 'obs_elevaion' in line:
                    namelist[3] = parse_a_line( line)
    #start again for the data
    with open(fin,'rt') as fh:
        dr = csv.DictReader( row for row in fh if not row.startswith('#') )
        for line in dr:
            if 'azimuth' in line:
                line_list = [line['datetime'],
                         float(line['altitude']),
                         float(line['azimuth']), 
                         [float(line['err_minus_altitude']),
                           float(line['err_plus_altitude'])],
                         [float(line['err_minus_azimuth']),
                           float(line['err_plus_azimuth'])] ]
            else:
                line_list = [line['datetime'],
                         float(line['x_image']),
                         float(line['y_image']), 
                         [float(line['err_minus_x_image']),
                           float(line['err_plus_x_image'])],
                         [float(line['err_minus_y_image']),
                           float(line['err_plus_y_image'])] ]
            if 'brightness' in line:
                line_list.extend( [float(line['brightness']),
                         [float(line['err_minus_brightness']),
                           float(line['err_plus_brightness'])] ] )
            else:
                line_list.extend( [100.0, [ 1.0,1.0]] )
            pts.append( line_list )
    return pts, namelist

def parse_a_line( linestr):
    """ given a string
    'key: value\n'
    return value only"""
    result = linestr.split(':')[1]
    result = result.strip( " {}\r\n'")
    return result
#@+node:martin.20150918192343.1: *3* writer
def write_event_file( fname, namestr, headstr, points):
    
    if fname.lower().endswith('txt'):
        return write_event_txt( fname, namestr, headstr, points)
    elif fname.lower().endswith('ecsv'):
        namelist = namestr.split(',')
        head1str = r"# %ECSV 0.9\n# ---\n# delimiter: ','\n# meta: !!omap\n"
        head1str += r"# - {camera_codename: " + namelist[0].strip('#') + "}\n"
        head1str += r"# - {obs_longitude: " + namelist[1] + "}\n"
        head1str += r"# - {obs_latitude: " + namelist[2] + "}\n"
        head1str += r"# - {obs_elevation: " + namelist[3] + "}\n"
        head1str = r"# datatype:\n"
        head1str += r"# - {name: datetime, unit: date_and_time, datatype: object}\n"

        return write_event_txt( fname,
                                head1str,
                                ecsv_build_headerstr(fname),
                                points)
    else:
        return False
#@+node:martin.20150918214232.1: *4* txt file
def write_event_txt( fname, head1str, head2str, points):
    """save a list of event points to a single txt file, return true/false
    points = [p1,p2,p3,...]
    p = [timestr, x, y, [dx-,dx+], [dy-,dy+], bright, [db-,db+] ]
    or maybe 
    p = [timestr, x, y, [dx-,dx+], [dy-,dy+], bright, [db-,db+], jd ]
    x, y can be floats, or can be strings"""
    logger = logging.getLogger()
    try:
        with open( fname, 'wt') as f:
            f.write( head1str)
            f.write( head2str)
            for p in points:
                p_string = ','.join( [str(p[0]), str(p[1]), 
                    str(p[2]), str(p[3][0]), str(p[3][1]), str(p[4][0]), str(p[4][1]),
                    str(p[5]), str(p[6][0]), str(p[6][1])] )
                if len(p) > 7: # julian day column exists
                    p_string = p_string + ', ' + str(p[7])
                f.write( p_string + '\n' )
    except (IOError,OSError):
        logger.warning('Failed to save event file, '+ fname)
        return False
    else:
        logger.debug('Saved event file, '+ fname)
        return True
#@+node:martin.20150918192428.1: *4* txt radec file
def write_radec_file( fname, namestr, headstr, points):
    """save a list of event points to a single txt file, return true/false"""
    #depreciated
    return write_event_txt( fname, namestr, headstr, points)
    
    # logger = logging.getLogger()    
    # try:
        # with open( fname, 'wt') as f:
            # f.write( namestr)
            # f.write( headstr)
            # for p in points:
                # f.write( p[0] + ', ' +
                    # str(p[1]) + ', ' +
                    # str(p[2]) + ', ' +
                    # '{0:.6E}'.format(p[3][0]) + ', ' +
                    # '{0:.6E}'.format(p[3][1]) + ', ' +
                    # '{0:.6E}'.format(p[4][0]) + ', ' +
                    # '{0:.6E}'.format(p[4][1]) + ', ' +
                    # '{0:.3E}'.format(p[5])  + ', ' +
                    # '{0:.3E}'.format(p[6][0]) + ', ' +
                    # '{0:.3E}'.format(p[6][1]) + '\n' )
    # except (IOError,OSError):
        # logger.warning('Failed to save event file, '+ fname)
        # return False
    # else:
        # logger.debug('Saved event file, '+ fname)
        # return True
#@+node:martin.20150919162042.1: *4* ecsv file
def ecsv_build_headerstr( fname):
    if 'pixel' in fname:
        head2str = r"# - {name: x_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: x_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_minus_x_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_x_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_minus_y_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_y_image, unit: pix, datatype: float64}\n"
    elif 'altazi' in fname:
        head2str = r"# - {name: alt_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: azi_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_minus_alt_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_alt_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_minus_azi_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_azi_image, unit: pix, datatype: float64}\n"
    elif 'radec' in fname:
        head2str = r"# - {name: RA_image, unit: pix, datatype: string}\n"
        head2str += r"# - {name: dec_image, unit: pix, datatype: string}\n"
        head2str += r"# - {name: err_minus_RA_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_RA_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_minus_dec_image, unit: pix, datatype: float64}\n"
        head2str += r"# - {name: err_plus_dec_image, unit: pix, datatype: float64}\n"
    else:
        raise Exception
    head2str += r"# - {name: brightness, unit: ADU, datatype: float64}\n"
    head2str += r"# - {name: err_minus_brightness, unit: ADU, datatype: float64}\n"
    head2str += r"# - {name: err_plus_brightness, unit: ADU, datatype: float64}\n"
    ##### jd column?
    
    if 'pixel' in fname:
        head2str += r"datetime,x_image,x_image,err_minus_x_image,err_plus_x_image,"
        head2str += r"err_minus_y_image,err_plus_y_image,"
    elif 'altazi' in fname:
        head2str += r"datetime,alt_image,azi_image,err_minus_alt_image,err_plus_alt_image,"
        head2str += r"err_minus_azi_image,err_plus_azi_image,"
    elif 'radec' in fname:
        head2str += r"datetime,RA_image,dec_image,err_minus_RA_image,err_plus_RA_image,"
        head2str += r"err_minus_dec_image,err_plus_dec_image,"
    else:
        raise Exception
    head2str += r"brightness,err_minus_brightness,err_plus_brightness,\n"
    return head2str
#@+node:martin.20131111100324.2315: ** data_path, make and check
def make_data_path( data_dir, today_date, secs = True, make = False):
    """build a data path, and return it using a given date. optional 
    creation of actual path, compared to version on remote cameras"""
    logger = logging.getLogger()
#    today_date = the_date.strftime('%Y-%m-%d')
    today_year, today_month, today_day = today_date.split('-')
    if secs:
        today_sec = datetime.datetime.now().strftime('%s')
        today_date = today_date + '_' + today_sec
    data_path = os.path.join( data_dir, today_year, today_month, today_date)
    #seconds on end as pseudo random to prevent directory errors
    logger.debug('dfns_make_data_path, ' + str(data_path) )
    if make:
        try:
            if os.path.isfile( data_path):
                os.remove( data_path)
            if not os.path.isdir( data_path):
                os.makedirs( data_path)
        except (IOError, OSError) as e:
            print('dfns_data_path_creation_error, ' + str(e) )
            return data_dir
    return data_path
#@+node:martin.20141111100951.2581: ** get file date from event filename
def date_from_fname( local_fname):
    """return 2014-12-24 format date for a given path to a standard event"""
    # /home/data/EVENTS/2015/06/2015-06-08/
    #    2015-06-08T10:41:10.0078_DFNSMALL20_DFNSMALL34 or
    #    2015-06-08T10-41-10.0078_DFNSMALL20_DFNSMALL34
    # but doesnt matter
    intermediate_date_str = os.path.basename(local_fname).split('_')[0]
     # 2015-06-08T10[:|-]41[:|-]10.0078
    return intermediate_date_str.split('T')[0] #2014-06-23
#@+node:martin.20141111100951.2580: ** time_from_fname
def time_from_fname( local_fname):
    logger = logging.getLogger()
    intermediate_time_str = os.path.basename(local_fname).split('.')[0] #00_2014-06-23_114029_DSC_0160
    exp_time_str = '_'.join( intermediate_time_str.split('_')[1:3] ) #2014-06-23_114029
    logger.debug( 'fname_extracted, ' + exp_time_str )
    tm_tuple = time.strptime( exp_time_str, '%Y-%m-%d_%H%M%S')
    logger.debug( 'fname_tuple, '  + str(tm_tuple) )
    exp_time_unix = calendar.timegm( tm_tuple )
    return exp_time_unix, exp_time_str
#@+node:martin.20130924130436.2253: ** combo records
def load_camera_group_list( fname):
    return read_log_file( fname)

def load_combo_log( local_dir):
    fname = os.path.join( local_dir, r'double_combo_log.txt')
    return read_log_file( fname)


# append to file in the day's data dir a list of cameras which this dir has
# been triangulated with.
def update_combo_log( local_dir, new_camera):
    fname = os.path.join( local_dir, r'double_combo_log.txt')
    return append_log_file( fname, new_camera)

def remove_all_combos(data_dir, date_range = []):
    """remove old double_combo_log.txt, so that triangulation can be re-run"""
    logger = logging.getLogger()
    logger.info( 'remove_all_combos, ' + str(data_dir) )
    for base_dir, dirs, files in os.walk( data_dir):
        #parse dir name, check if in date_range
        # eg /home/data/DFNSMALL06/2014/01/2014-01-09_DFNSMALL06_1389258901
        logger.debug( 'removing_old_combo, ' + str(base_dir) )
        remove_a_combo( base_dir, date_range)
    return 
        
def remove_a_combo( local_dir, date_range = []):
    """remove a single combo log file, given directory"""
    logger = logging.getLogger()
    comb_file = os.path.join( local_dir, 'double_combo_log.txt')
    if os.path.isfile( comb_file):
        logger.debug('combo_found, ' + comb_file)
        if date_range == [] or os.path.basename( local_dir)[:10] in date_range:
            try:
                os.remove( comb_file)
                logger.debug('combo_removed, ' + comb_file)
                return True
            except OSError as e:
                logger.info('combo_remove_fail, ' + str(e))
                pass

#@+node:martin.20140922115550.2513: ** fireball records
def write_fireball_records( records, local_base_dir):
    """write a list or ordered_dict of fireball_records as separate files
    to local_dir"""
    logger = logging.getLogger()
    local_dir = os.path.abspath( os.path.realpath( local_base_dir) )
    tally = True
    for rcd in records:
        # construct full path
        day = date_from_fname( rcd['event_file'] ) #2014-12-24
        year, month = day.split('-')[0:2]
        local_dir = os.path.join( local_base_dir, year, month, day)
        #make folders if necessary
        if not os.path.exists( local_dir):
            logger.info( 'dfns_fireball_make_dir, ' + local_dir )
            os.makedirs( local_dir)
        tally = tally and write_fireball_record( rcd, local_dir)
    return tally

def read_fireball_records( local_dir):
    """return a list of all dicts of fireball_records present in a local_dir"""
    logger = logging.getLogger()
    local_dir = os.path.abspath( os.path.realpath( local_dir) )
    logger.info( 'read_fireball_records_path, ' + local_dir)
    records = []
    files = glob.glob( os.path.join( local_dir, '*fireball.cfg'))
    for fle in files:
        logger.debug( 'fireball_record_found, ' + str(fle) )
        records.append( read_fireball_record( os.path.join( local_dir, fle) ))
    logger.debug( 'fireball_records_loaded, ' + str(records) )
    return records
#@+node:martin.20150428144920.5: *3* single records
def read_fireball_record( fname):
    """return a dict of fireball parameters from fname"""
    cfg = configparser.ConfigParser()
    cfg.read( fname)
    record = {}
    for item in cfg['fireball']:
        record[item] = cfg['fireball'][item]
    return record

    
def write_fireball_record( record, local_dir):
    """given a local dir and record_dict, write as a INI file
    format using event_file key as filename + _fireball.cfg"""
    local_dir = os.path.abspath( os.path.realpath( local_dir) )
    cfg = configparser.ConfigParser()
    cfg['fireball'] = {}
    cfg['cameras'] = {}
    for item in record:
        #exclude overkill field, as this is temp for handling on 1 day.
        if 'email_owner' in item or 'config_file' in item:
            cfg['cameras'][item] = str( record[item] )
        else:
            if not 'overkill' in item:
                cfg['fireball'][item] = str( record[item] )
    try:
        with open( os.path.join( local_dir, record['event_file']+ '_fireball.cfg'),
                   'wt') as configfile:
            cfg.write( configfile)
    except IOError:
        return False
    else:
        return True
#@+node:martin.20151113223820.1: ** read meteorite cfg file
def read_fireball_cfg( fname):
    
    met_config = configparser.ConfigParser( interpolation = None)
    met_config.read( fname)
    # convert met_dict to a real dict 
    met_dict = {}
    for section in met_config.sections():
        met_dict[ section] = {}
        for key, val in met_config.items(section):
            met_dict[section][key] = val
    return met_dict
    
#@+node:martin.20151113223320.1: ** write meteorite cfg file
def write_fireball_cfg( fname, conf_dict):
    
    logger = logging.getLogger()
    new_conf = configparser.ConfigParser( interpolation = None)
    #copy from a real dict to a configparser 'dict'
    for sec in conf_dict:
        #print('sec, ', sec)
        new_conf[str(sec)] = {}
        for item in conf_dict[str(sec)]:
            dummy = conf_dict[str(sec)][str(item)]
            new_conf[str(sec)][str(item)] = str(dummy)
    try:
        with open(fname, 'wt') as configfile:
            new_conf.write( configfile)
            configfile.flush()
    except IOError as e:
        logger.warning('Failed_to_save_new_cfg_file, '+ fname + ', ' + str(e) )
        return False
    else:
        logger.debug('Saved_new_cfg_file, '+ fname)
        return True
#@+node:martin.20131204145251.2654: ** trajectory data files
#@+node:martin.20150309095131.4: *3* load split
def load_split_traj_file( fname):
    """given a traj file, load it and return lists of data for columns
    format could be old style csv file, or astropy ecsv
    """
    logger = logging.getLogger()
    logger.debug('split_trajectory_file_called, ' + str(fname))
    if not os.path.isfile( fname):
        print( 'traj_file_not_exist, ' + fname)
        raise
    if fname.endswith('.ecsv'):
        time, lat, lon, elev, x,y,z, brightness = load_traj_ecsv( fname)
    elif 'MOP' in fname: #new py style
        time, lat, lon, elev, x,y,z, brightness = load_traj_MOP( fname)
    else: #old idl style
        time, lat, lon, elev, brightness = [],[],[],[],[]
        x,y,z = [],[],[]
        for item in load_traj_file( fname):
            dumvar = item.split(',')
            if len(dumvar) == 11: #correct number of fields
                time.append( str(dumvar[0])) #iso string
                lat.append( float(dumvar[1])) #in deg
                lon.append( float(dumvar[2])) # in deg
                elev.append( float(dumvar[3])) #in m
                x.append( float(dumvar[4])) #in km
                y.append( float(dumvar[5])) #in km
                z.append( float(dumvar[6])) #in km
                brightness.append( float(dumvar[10])) # float
        logger.debug('split_traj_finished, ' + str(fname))
        # 2 sets of data from 2 cameras here
        # try globally sorting by time
        lat.sort( key=dict(zip(lat,time)).get)
        lon.sort( key=dict(zip(lon,time)).get)
        elev.sort( key=dict(zip(elev,time)).get)
        x.sort( key=dict(zip(x,time)).get)
        y.sort( key=dict(zip(y,time)).get)
        z.sort( key=dict(zip(z,time)).get)
        brightness.sort( key=dict(zip(brightness,time)).get)
        time.sort() #keep time as str
    return time, lat, lon, elev, x,y,z, brightness
#@+node:martin.20150309095131.3: *3* load
def load_traj_file( fname):
    """load traj file and return one big string list of data
    ignore comments
    (called by load_split_traj_file() )"""
    logger = logging.getLogger()
    logger.debug('load_trajectory_file_called, ' + str(fname))
    traj_list = []
    try:
        with open( fname, 'rt') as f:
            for dumvar in f.readlines():
                #print( 'dumvare', dumvar)
                if not dumvar.rstrip('\n\t').startswith('#'):
                    traj_list.append( dumvar.rstrip('\n\t') )
    except (IOError, OSError) as e:
        #print('traj_read_error, ' + str(e))
        logger.warning('traj_read_error, ' + str(e))
    logger.debug('traj_file_loaded, ' + str(len(traj_list)) )
    #print('trajl,', traj_list)
    return traj_list
#@+node:martin.20180227110835.1: *3* load ecsv
def load_traj_ecsv( fname):
    """load an astropy type trajectory file, called by load_split
    its not a real ecsv, as its 2 ecsv files cat'd and welded together"""
    
    #read in text file
    tab1 = []
#    tab2 = []
    with open(fname,'rt') as fh:
        tab1.append( fh.readline())
        flag = True
        for dat in fh.readlines():
            if flag:
                tab1.append( dat)
            if 'ecsv' in dat: #header line siwth to 2nd table
                flag = False
    #load each one as Table
    
    #join table columns together
    
    data_table = Table.read( fname, delimiter = ',',
                             format='ascii.ecsv',
                             guess=False )
    
    time = data_table['datetime']
    lat = data_table['lat']
    lon = data_table['lon']
    elev = data_table['elev']
    x = data_table['x']
    y = data_table['y']
    z = data_table['z']
    brightness = data_table['brightness']
    return time, lat, lon, elev, x,y,z, brightness
#@+node:martin.20180724152405.1: *3* load MOP
def load_traj_MOP( fname):
    """load an MOP trajectory file, called by load_split
    its got 2 tables each with header names and unordered columns"""
    time, lat, lon, elev, brightness = [],[],[],[],[]
    x,y,z = [],[],[]
    #load all the comments and find the event names
    tab1 = []
    tab2 = []
    with open( fname, 'rt') as f:
        ef1 = f.readline()
        ef2 = f.readline()
        loc1 = f.readline()
        header1 = f.readline().split(',')
        while True:
            dat = f.readline()
            if dat.startswith('#'):
                break
            tab1.append( dat)
        loc2 = dat
        header2 = f.readline().split(',')
        while True:
            dat = f.readline()
            if not dat or dat.startswith('#'):
                break
            tab2.append( dat)
    #extract out named columns
    header1[0] = header1[0].lstrip('#')
    header2[0] = header2[0].lstrip('#')
    header1 = [a.strip() for a in header1]
    header2 = [a.strip() for a in header2]
    tabd1 = []
    for row in tab1:
        dd = {}
        dat = row.split(',')
        dat = [a.strip() for a in dat]
        for b in range(len(dat)):
            dd[ header1[b]] = dat[b]
        tabd1.append( dd)
    tabd2 = []
    for row in tab2:
        dd = {}
        dat = row.split(',')
        dat = [a.strip() for a in dat]
        for b in range(len(dat)):
            dd[ header2[b]] = dat[b]
        tabd2.append( dd)
    #tabd = list of dicts
    for tab in (tabd1,tabd2):
        for row in tab:
            #print(row)
            time.append( row['datetime']) #iso string
            lat.append( row['latitude']) #in deg
            lon.append( row['longitude']) # in deg
            elev.append( float(row['height'])) #in m
            x.append( float( row['X_geo'])) #in km
            y.append( float( row['Y_geo'])) #in km
            z.append( float( row['Z_geo'])) #in km
            if 'brightness' in row:
                brightness.append( row['brightness']) # float
            else:
                brightness.append( 255.0 ) # float
    logger.debug('split_traj_finished, ' + str(fname))
    # 2 sets of data from 2 cameras here
    # try globally sorting by time
    lat.sort( key=dict(zip(lat,time)).get)
    lon.sort( key=dict(zip(lon,time)).get)
    elev.sort( key=dict(zip(elev,time)).get)
    x.sort( key=dict(zip(x,time)).get)
    y.sort( key=dict(zip(y,time)).get)
    z.sort( key=dict(zip(z,time)).get)
    brightness.sort( key=dict(zip(brightness,time)).get)
    time.sort() #keep time as str
    return time, lat, lon, elev, x,y,z, brightness
#@+node:martin.20150309095131.5: *3* events from a traj file
def events_from_traj_file( local_fname):
    """given a traj file of a triangulated fireball, return the event files
    used for the triangulation"""
    logger = logging.getLogger()
    event_files = []
    if local_fname.endswith('.ecsv'):
        raise NotImplementedError
    else:
        with open( local_fname, 'rt') as f:
            for a in range(5):
                line = f.readline().strip()
                if line.startswith('#event'):
                    event_files.append( line.split(',')[1].strip() )
    logger.debug( 'traj_files, ' + str(event_files))
    return event_files
#@+node:martin.20150309095131.6: *3* time from traj file name
def time_from_traj_fname( local_fname):
    """given a traj file, find unix event time"""
    logger = logging.getLogger()
    # 2015-03-02T21:31:21.0097_DFNSMALL16_DFNSMALL22_trajectory.csv
    intermediate_time_str = os.path.basename(local_fname).split('.')[0]
    #2015-03-02T21:31:21.0097_DFNSMALL16_DFNSMALL22_trajectory
    exp_time_str = intermediate_time_str.split('_')[0] #2015-03-02T21:31:21.0097
    logger.debug( 'time_from_traj_fname_extracted, ' + exp_time_str )
    #could be : (old style) or - (new style) as separator
    try:
        tm_tuple = time.strptime( exp_time_str, '%Y-%m-%dT%H-%M-%S')
    except ValueError:
        tm_tuple = time.strptime( exp_time_str, '%Y-%m-%dT%H:%M:%S')
    logger.debug( 'time_from_traj_fname_tuple, '  + str(tm_tuple) )
    exp_time_unix = calendar.timegm( tm_tuple )
    return exp_time_unix
#@+node:martin.20150428144920.3: ** trajectory handling
def fireball_velocity( jday_timearr, x_arr, y_arr, z_arr):
    """given time, x,y z, return velocity array.
    vel[0] = -1.0
    arrays are assumed sorted already"""
    time_arr = [jd_to_sec(a) for a in jday_timearr]
    vel = [-1.0]
    for i in range(1, len(time_arr)):
        vel.append( distance_between_points( x_arr[i],y_arr[i],z_arr[i],
                                            x_arr[i-1],y_arr[i-1],z_arr[i-1]) / 
                                            (time_arr[i] - time_arr[i-1]) )
    return vel
#@+node:martin.20130924130436.2252: ** general file handling
def append_log_file( fname, text_to_add):
    logger.debug('log_append_called, ' + str(fname) + ', ' + text_to_add)
    try:
        with open( fname, 'at') as f:
            if text_to_add.rstrip() != '':
                f.write( text_to_add.strip() + '\n')
    except (IOError, OSError) as e:
        logger.warning('log_append_error, ' + str(e))
        return False
    return True
    
def read_log_file( fname):
    logger.debug( 'log_read_called, ' +str(fname) )
    log_list = []
    if os.path.exists( fname):
        try:
            with open( fname, 'rt') as f:
                for dumvar in f.readlines():
                    if dumvar.strip('\n\t') != '':
                        if not dumvar.strip('\n\t').startswith('#'):
                            log_list.append( dumvar.strip('\n\t') )
        except (IOError, OSError) as e:
            logger.warning('log_read_error, ' + str(e))
    return log_list
#@+node:martin.20131204145251.3042: ** general maths
def distance_between_points( x1, y1, z1, x2, y2, z2):
    return math.sqrt( pow((x1-x2),2)
                    + pow((y1-y2),2)
                    + pow((z1-z2),2) )

def jd_to_sec( jd):
    """convert fractional julian day to seconds, ignoring the whole number"""
    jd_f = float( jd)
    return 86400.0 * (jd_f - int(jd_f))
#@-others
#@-leo

from os import listdir
from os.path import isfile,isdir, join
from posixpath import splitext
from typing import List, Sequence, Tuple
import pandas as pd
import numpy as np
from .GPS_GPGGA import GPGGAParser


"""
Utility functions for the purpose of manipulating metadata and file system names
"""


def get_hsi_folder_names(path_RootFolder, not_std = None,filter_out=None):
    """ Returns the name or 'id' of each hsi folder within a root directory """
    if not_std is None:
        if filter_out is None:
            return [x for x in listdir(path_RootFolder)]
        else:
            return [x for x in listdir(path_RootFolder) if x[-len(filter_out):] !=filter_out]
    elif not_std:
        return [x for x in listdir(path_RootFolder) if x[-3:] != 'STD']

def get_folder_list_from_root_dir(path_RootFolder, filter_out = None, not_std = None):
    """ Takes the root directory of spectral imagery, returns the array of full path folders. Assumes std identification work is done when not_std is used. Optional filter param: last three characters of folder """
    if filter_out is None and not_std is None:
        return [join(path_RootFolder,x) for x in listdir(path_RootFolder) if not(isfile(join(path_RootFolder,x)))]
    elif not_std: #The calibration standards are not of interest
        return [join(path_RootFolder,x) for x in listdir(path_RootFolder) if 'std' not in x.lower()]
    else : # grab the calibration standards only/ or a certain file extension
        return [join(path_RootFolder,x) for x in listdir(path_RootFolder) if x[-len(filter_out):] == filter_out]

def get_capture_dirs(path_RootFolder:str):
    return [join(path_RootFolder,x) for x in listdir(path_RootFolder) if 'capture' in x]


def get_absolute_path_HDR_from_parent_path_and_exposure(path_parent,exposure = None):
    """ Takes the parent folder(HSI folder) and select exposure, returns the absolute path to the HDR of that exposure. No exposure? first found subdirectory """
    
    if exposure is None: #unless no tripod image exists, this will always return a path, 10ms by default alphabetical order
        path_parent_exposure = [join(path_parent,x) for x in listdir(path_parent) if isdir(join(path_parent,x))and x[-3:] !=".db" ][0]
        path_parent_exposure_subdir = [join(path_parent_exposure,x) for x in listdir(path_parent_exposure) if isdir(join(path_parent_exposure,x))][0]
        path_hdr = [join(path_parent_exposure_subdir,x) for x in listdir(path_parent_exposure_subdir) if x.find('.hdr') != -1][0]
        return path_hdr

    else: 
        # print(f"searching for {exposure}")
        found = False
        #
        ## Look for the exposure by selection
        #
        for x in listdir(path_parent):
            if x.find(exposure) != -1:
                path_parent_exposure = join(path_parent,x)
                found = True
                # print(f"found exposure path{path_parent_exposure}")
                break
            else:
                found = False
        if found:
            for x in listdir(path_parent_exposure):
                if isdir(join(path_parent_exposure,x)):
                    path_parent_exposure_subdir = join(path_parent_exposure,x)
                    found = True
                    # print(f"found subdirectory: {path_parent_exposure_subdir}")
                    break
        else:
            return None
        #
        ## The exposure has been found
        #
        found = False
        for x in listdir(path_parent_exposure_subdir):
            if x.find('.hdr') != -1:
                path_hdr = join(path_parent_exposure_subdir,x)
                found = True
                break    
        if found:
            return path_hdr
        else:
            return None

def get_mask_from_root_dir(path_RootFolder,ext_filetype):
    """" Utility to retrieve a multi-class mask from sub directory of parent spectral image folder """
    return [join(path_RootFolder,x) for x in listdir(path_RootFolder) if x[-3:] == ext_filetype][0]

def get_hdr_file_path(path):
    """ Retrieves the path to find a header file of an image set, must be have exposure sub directories in path already"""
    hdrFilename = [f for f in listdir(path) if f[-3:] == 'hdr'][0]
    return join(path,hdrFilename)

def get_hdr_contents(hdrPath:str):
    """" Opens the hdr, or any other text type, file and returns all lines """
    hdr = open(hdrPath,'r') #open the header file
    return hdr.readlines()

def retrieve_decimal_degree_array(gps):
    """ Manipulate a string to produce an array of decimal degree values """
    # log output
    gpsSplit = gps[23].split('=')
    gpsSplit = gpsSplit[1].split('{')[1]
    gpsSplit = gpsSplit.split('}')[0]
    gpsSplit = gpsSplit.split('$GNRMC,')[1:]

    return gpsSplit

def grab_raw_coords(header):
    """ This is the main control over utility functions, return value is large array of dd points """
    gpsInfo = get_hdr_contents(header)#hdr.readlines() # find the unformatted gps info
    gpsDD = retrieve_decimal_degree_array(gpsInfo) # DD coordinate array
    return gpsDD

def dm(x):
    """ UTILITY: Takes a string in dddmmmm format and breaks it apart """
    degrees = int(x) // 100
    minutes = x - 100*degrees
    return degrees, minutes

def decimal_degrees(degrees, minutes):
    """ UTILITY: Converts decimal degress to coordinate point """
    return degrees + minutes/60 

def extract_dd_latitude(dd):
    """ UTILITY: Pulls latitude decimal degree characters from raw string """
    ddItems = dd.split(',')
    return ddItems[2]


def extract_dd_longitude(dd):
    """ UTILITY: Pulls longitude decimal degree characters from raw string """
    ddItems = dd.split(',')
    return ddItems[4]


def extract_coordinate(rawCoord):
    """ Takes in one raw unformatted data point and returns lat/long """
    try:

        ddLat = dm(float(extract_dd_latitude(rawCoord)))
        ddLong = dm(float(extract_dd_longitude(rawCoord)))
        degLat = decimal_degrees(ddLat[0],ddLat[1])
        degLong = decimal_degrees(ddLong[0],ddLong[1])
        return (degLat, degLong)
    except Exception as e:
        print(e)

def get_time(path):
    """Grabs and formats the timestamp within header file"""
    hdrInfo = get_hdr_contents(path)
    timestamp = hdrInfo[12].split('=')[1]
    timestamp = timestamp.split("\n")[0].strip()
    timestamp = pd.Timestamp(timestamp[0:19])#,"%Y-%m-%dT%H:%M:%S") 
    return timestamp.to_pydatetime()


def get_solar_irradiance_from_hdr_path(path):
    list_values = get_hdr_contents(path)
    irradiance = str(list_values[16].split('='))
    irradiance = np.array(irradiance.split('{')[1].split('}')[0].split(','))
    return irradiance

def get_gain_array(path):
    """Grab the data gain values as an array"""
    hdrInfo = get_hdr_contents(path)
    gains = hdrInfo[15]
    gains = gains.split('{')[1]
    gains = gains.split('}')[0]
    gains = np.array(gains.split(',')).astype(np.float32)
    return gains

def get_gpgga_from_nav(path:str) -> Tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Given a .NAV file of GPGGA coordinates, parses through the lead lines to identify $GPGGA and extracts data for each one
    
    Returns:
    --------
        latList : Latitude list as np.ndarry
        longList : Longitude list as np.ndarry 
    """
    if isfile(path) and path[-3:] == "nav":
        gpsList = get_hdr_contents(path)
        return extract_ll_lists(gpsList)
    else : 
        raise FileNotFoundError("NAV file does not exist")

def extract_ll_lists(gpsList:List) -> Sequence[np.ndarray]:
    """
    UTILITY to formulate the relevant info from GPGGA text lines

    Returns:
    --------
        latList : Latitude list as np.ndarry
        longList : Longitude list as np.ndarry 
    """
    gpggaList = [coord for coord in gpsList if coord.find("GPGGA")!=-1] # raw list
    gpsList = [GPGGAParser(coord) for coord in gpggaList]  # Parsed as objects
    latLongAlt = [(gps.latitude,gps.longitude,gps.altitude) for gps in gpsList] # LLA tuple (lat,lon,alt)
    latList = [lat[0] for lat in latLongAlt]
    longList = [long[1] for long in latLongAlt]

    return latList, longList

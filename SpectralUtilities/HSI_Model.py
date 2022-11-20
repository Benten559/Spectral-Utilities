from typing import Union
import os
import math
from os.path import isfile, isdir, join
from spectral import *
import spectral.io.envi as envi
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import geopy
import geopy.distance
import tifffile
import pandas as pd
from PIL import Image
from osgeo import gdal, osr
from . import HDRprocess


def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


class HSI_Model:
    """ An object designed to house: Hypercube, wavelengths used, RGB image, logical mask """

    def __init__(self, path_hcube: str, imgName: str, path_mask: str = None, dataset: str = None, norm_max: int = None, nav_file:str = None, load_data:bool = False) -> None:
        """ 
        Properties will contain all relevant meta-data for HSI image. 

        Parameters:
            path_hcube -- Absolute path to HDR file, assumes DAT file is within same directory
            imgName -- Camera given name of this HSI,
            path_mask -- Optional path to associated segmentation mask for this image, or set to 'self'
            dataset -- Only options are 'berry' and 'tripod'. Defaults to tripod, optional parameter for rgb creation
            norm_max -- The max value given to the rgb composite of this image
        """

        #
        ## Camera generated data
        #
        self.imageName = None
        self.hcube = None
        self.wv = None
        self.timeTaken = None
        self.gain = None
        self.maskModified = False
        self.stdRating = None

        #
        ## Calculated properties
        #
        self.kernelCenter = None
        self.sharp = None
        self.rgb = None
        self.mask = None
        self.stdMask = None
        self.stdAvgRad = None
        self.vineAvgRad = None

        print("loading hypercube...")
        envi_obj = envi.open(path_hcube)
        if load_data:
            envi_obj = envi_obj.load()
        self.wv = envi_obj.bands.centers
        self.imageName = imgName
        
        #setting an rgb
        if dataset is None:
            print(f"RGB is not initialized")
        elif dataset == "berry" or dataset == "leaf":
            self.stdRating = .99
            print(
                f"constructing composite image using wavelengths:{self.wv[2]}nm, {self.wv[10]}nm, {self.wv[30]}nm ")
            # if norm_max is None: # default will be [0-1]
            r = norm(envi_obj[:, :, 2])
            g = norm(envi_obj[:, :, 10])
            b = norm(envi_obj[:, :, 30])
            self.rgb = cv2.merge([r, g, b])        
        
        #normalizing the rgb
        if (not norm_max is None) and (self.rgb is not None) : 
            self.rgb = cv2.normalize(self.rgb, None, alpha=1, beta=norm_max,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        # Assign properties:
        if load_data:
            self.hcube = np.array(envi_obj)
        else:
            self.hcube = envi_obj

        # assigning a mask, by file or by rgb data
        if path_mask != None:
            if path_mask == "self":
                self.set_mask_by_threshold()
            else:
                self.load_mask(path_mask)

        # pushbroom data has no gain data, TODO get time from new header
        if (dataset is not None) and (dataset != 'pushbroom'):
            self.timeTaken = HDRprocess.get_time(path_hcube)
            self.gain = HDRprocess.get_gain_array(path_hcube)
        
        # assign gps from NAV file
        if nav_file is not None:
            self.latList , self.longList = HDRprocess.get_gpgga_from_nav(nav_file)
        else:
            try:
                rawGPS = HDRprocess.grab_raw_coords(path_hcube)
                # For consistency, these single points will be named list, but otherwise its just one Lat/Long
                self.latList = np.mean([HDRprocess.extract_coordinate(x)[0] for x in rawGPS])
                self.longList = np.mean([HDRprocess.extract_coordinate(x)[1]*-1 for x in rawGPS])
            except Exception as e:
                print(f"Error loading coordinates: \n{e}")
        print("HSI load complete...")

    def load_entire_image_data(self) -> None:
        """
        By default, no data is loaded into memory so that a memmap may be used. 
        Commonly needed with BIL pushbroom files
        If this is not necessary, call this method to conduct analysis on "hcube".  
        """
        assert isinstance(self.hcube,spectral.io.bilfile.BilFile) 
        self.hcube = self.hcube.load()
    def load_mask(self, path_mask):
        """ Called on initialization, if a path to multi-class mask is provided to constructor. """
        if isfile(path_mask):
            mask_multiclass = cv2.imread(path_mask)
            mask_multiclass = cv2.cvtColor(mask_multiclass, cv2.COLOR_BGR2RGB)
            self.mask = (mask_multiclass[:, :, 2] == mask_multiclass[:, :, 1]).astype(
                np.int8) * 255

    def load_mask_from_model_output(self, maskObj):
        self.mask = maskObj

    def save_rgb(self, path_save_dir):
        """ Used to save the composite rgb image of HSI """
        savePath = join(path_save_dir,self.imageName)
        plt.imsave(f"{savePath}.png", self.rgb)
        print(f"RGB saved to: {savePath}")


    def save_mask(self, path_save_dir,which="vine"):
        assert isdir(path_save_dir)

        if which == "vine" and not self.mask is None:
            cv2.imwrite(f"{join(path_save_dir,self.imageName)}.png", self.mask*255)
        elif which == "std" and not self.stdMask is None:
            cv2.imwrite(f"{join(path_save_dir,self.imageName)}.png",self.stdMask*255)

    def save_rgb_sxs_mask(self, path_save_dir,extent = (0,1024,0,1024)):
        """
        Choose a path to save a side-by-side image, useful for validation.
        This assumes a 1 mega-pixel data image. Optional parameter extent to provide a data crop 
        Extent: (row_min , row_max, column_min, column_max)
        """
        assert isdir(path_save_dir)
        assert not self.rgb is None
        assert not self.mask is None

        rgb = self.rgb[extent[0]:extent[1],extent[2]:extent[3],1]
        mask = self.mask[extent[0]:extent[1],extent[2]:extent[3]]
        plt.imsave(f"{join(path_save_dir,self.imageName)}.png",
                   np.hstack([rgb, mask*255]))

    def save_band_sxs_std_mask(self, path_save_dir):
        assert isdir(path_save_dir)
        assert not self.rgb is None
        assert not self.stdMask is None
        plt.imsave(f"{join(path_save_dir,self.imageName)}.png",
                   np.hstack([self.rgb[:, :, 1], self.stdMask]))

    def set_rgb_by_wv_index(self, r: int, g: int, b: int, norm_max: int = None,rot : bool = False) -> None:
        """Reset the auto-generated rgb manually by selecting the index of wavelengths, if no norm_max is specified then pixels will be decimals"""
        r = norm(self.hcube[:, :, r])
        g = norm(self.hcube[:, :, g])
        b = norm(self.hcube[:, :, b])
        if rot:
            self.rgb = np.rot90(cv2.merge([r,g,b]))
        else:
            self.rgb = cv2.merge([r, g, b])
        if not norm_max is None:
            self.rgb = cv2.normalize(
                self.rgb, None, alpha=1, beta=norm_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    def remove_std_from_mask(self,xPixelOffset=0, yPixelOffset: int = 500):
        """
        Assuming a circular standard in the bottom half of an image, this will remove those pixels (set them to 0) within the existing logical mask
        If the location of the standard is higher/lower on the frame, this new mask can be adjusted using the yPixelOffset parameter
        Should there be multiple modifications to the mask using this function, it will reset the original mask everytime (useful for trial and error)
        """
        if self.mask is None or self.sharp is None:
            print("Load a mask prior to trying to remove circular artifacts")
            return
        elif not self.kernelCenter is None:
            try:
                self.set_mask_by_threshold(kernel_center=self.kernelCenter)

                mask_RemoveCalibrationStd = np.ones(
                    (self.hcube.shape[0], self.hcube.shape[1]))
                trim_offsetY = yPixelOffset
                trim_offsetX = xPixelOffset
                circles = cv2.HoughCircles(
                    self.sharp[trim_offsetY:, trim_offsetX:], cv2.HOUGH_GRADIENT, 1, 32, param1=1, param2=24, minRadius=30, maxRadius=40)
                circles = np.uint16(np.around(circles))
                stdLoc = circles[0, :][0]

                mask_RemoveCalibrationStd = cv2.circle(img=mask_RemoveCalibrationStd, center=(
                    stdLoc[0]+trim_offsetX, stdLoc[1]+trim_offsetY), radius=55, color=(0, 0, 0), thickness=-1).astype(np.uint8)

                self.mask = cv2.bitwise_and(
                    self.mask, self.mask, mask=mask_RemoveCalibrationStd)
            except Exception as e:
                raise Exception(f"Error occured during removal of circular standard\n {e}")
    def set_circular_std_pixel_mask(self, xPixelOffset:int = 0, yPixelOffset: int = 500):
        """
        Retains the pixel-location of the circular standard as a logical mask.
        Assumes dark-cabinet lab data and that it is in the focal view somewhere. 
        Should there be multiple modifications to the mask using this function, it will reset the original mask everytime (useful for trial and error)
        Until more sophisticated checks are developed, make sure the location is accurate, PLEASE.
        """
        if self.mask is None or self.sharp is None:
            print("Load a mask prior to trying to remove circular artifacts")
            return
        elif not self.kernelCenter is None:
            self.set_mask_by_threshold(kernel_center=self.kernelCenter)

        mask_KeepCalibrationStd = np.zeros((1024, 1024))
        trim_offsetY = yPixelOffset
        trim_offsetX = xPixelOffset
        circles = cv2.HoughCircles(
            self.sharp[trim_offsetY:, trim_offsetX:], cv2.HOUGH_GRADIENT, 1, 32, param1=1, param2=24, minRadius=30, maxRadius=40)
        circles = np.uint16(np.around(circles))
        stdLoc = circles[0, :][0]
        mask_KeepCalibrationStd = cv2.circle(img=mask_KeepCalibrationStd, center=(
            stdLoc[0]+trim_offsetX, stdLoc[1]+trim_offsetY), radius=stdLoc[2], color=(255, 255, 255), thickness=-1).astype(np.uint8)

        self.stdMask = mask_KeepCalibrationStd

    def set_mask_by_threshold(self, kernel_center=8):
        """
        Using the currently set rgb composite, apply a sharpened thresholding over normalized pixels 
        create a segmented mask. 
        Two options for tweaking results would be modifying the wavelengths used for rgb and choosing a 
        different kernel intensity value. 
        The chosen kernel center will be saved
        """
        self.kernelCenter = kernel_center
        kernel = np.array([[0, -1, 0],
                           [-1, kernel_center, -1],
                           [0, -1, 0]])
        img_sharp = cv2.filter2D(self.rgb, ddepth=-1, kernel=kernel)
        # Grayscale image
        img_gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
        sharp, thresh = cv2.threshold(
            img_gray.astype(np.uint8), 0, 1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.mask = thresh.astype(np.uint8)
        self.sharp = img_gray

    def roi_equalization(image):
        """
        In progress... normalize image pixels with a region of interest
        """
        ROI = image[400:800, 600:800]

    #  Calculate mean and STD
        mean, STD = cv2.meanStdDev(ROI)

    #  Clip frame to lower and upper STD
        offset = 0.2
        clipped = np.clip(image, mean - offset*STD, mean +
                          offset*STD).astype(np.uint8)

# Normalize to range
        result = cv2.normalize(clipped, clipped, 0, 255,
                               norm_type=cv2.NORM_MINMAX)
        return result

    def histogram_equalization(img_in):
        """
        In progress... normalize image pixels 
        """
    # segregate color streams
        b, g, r = cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
        cdf_b = np.cumsum(h_b)
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
        cdf_m_b = np.ma.masked_equal(cdf_b, 0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / \
            (cdf_m_b.max() - cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g, 0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / \
            (cdf_m_g.max() - cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

        cdf_m_r = np.ma.masked_equal(cdf_r, 0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / \
            (cdf_m_r.max() - cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))
    # validation
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))
        return img_out

    def set_all_masked_bands_radiance_array(self):
        assert not self.hcube is None and not self.gain is None and self.hcube.shape[2] == len(
            self.wv) and not self.mask is None
        radiance = []
        for band in range(len(self.wv)):
            bandImg = self.hcube[:, :, band]
            maskedBand = cv2.bitwise_and(bandImg, bandImg, mask=self.mask)
            maskedRadBand = maskedBand * self.gain[band]

            numPixels = len(maskedRadBand[maskedRadBand != 0])
            sumPixels = sum(maskedRadBand[maskedRadBand != 0])

            if self.gain[band] == 0:
                avg = 0
            else:
                avg = sumPixels/numPixels
            radiance.append(avg)
        self.vineAvgRad = np.array(radiance)

    def set_std_rad_avg(self):
        """
        Store the average radiance of a circular standard within property: 
        """
        assert not self.hcube is None and not self.gain is None and self.hcube.shape[2] == len(
            self.wv) and not self.stdMask is None

        radiance = []
        for band in range(len(self.wv)):
            bandImg = self.hcube[:, :, band]
            maskedBand = cv2.bitwise_and(bandImg, bandImg, mask=self.stdMask)
            maskedRadBand = np.multiply(maskedBand, self.gain[band])

            numPixels = len(maskedRadBand[maskedRadBand != 0])
            sumPixels = sum(maskedRadBand[maskedRadBand != 0])

            if self.gain[band] == 0:
                avg = 0
            else:
                avg = sumPixels/numPixels
            radiance.append(avg)
        self.stdAvgRad = np.array(radiance)

    def get_segmented_avg_reflectance_array(self, stdRating: float, zeroOffset=0):
        """
        Pass in the reflectance rating of the std being used to calculate.
        If some bands contained zero'd values specify the offset to avoid division by zero.
        """
        assert not self.hcube is None
        assert not self.mask is None
        assert not self.stdMask is None
        assert not self.stdAvgRad is None
        assert not self.vineAvgRad is None
        assert stdRating > 0
        assert zeroOffset < len(self.wv)

        return [self.vineAvgRad[i]*stdRating / self.stdAvgRad[i] for i in range(zeroOffset, len(self.wv))]

    def save_reflectance_segmented_cube_as_h5(self, save_path, stdRating: float = .99, zeroOffset: int = 0,extent = (0,1024,0,1024)) -> bool:
        """
        Saves a hypercube after application of pixels to reflectance
        """
        assert not self.hcube is None
        assert not self.mask is None
        assert not self.stdMask is None
        assert isdir(save_path)
        assert stdRating > 0

        reflCube = []
        for band in range(zeroOffset, len(self.wv)):
            bandImg = self.hcube[extent[0]:extent[1], extent[2]:extent[3], band]
            maskedBand = cv2.bitwise_and(bandImg, bandImg, mask=self.mask[extent[0]:extent[1], extent[2]:extent[3]])
            maskedRadBand = np.multiply(np.multiply(
                maskedBand, self.gain[band]), stdRating)
            reflBand = np.divide(maskedRadBand, self.stdAvgRad[band])
            reflCube.append(reflBand)
        try:
            with h5py.File(f"{join(save_path,self.imageName)}.hdf5", "w") as f:
                f.create_dataset("img", data=reflCube)
            return True
        except Exception as e:
            print(f"An issue occured while writing data to H5: {e}")
            return False

    def georef_band(self, savePath:str,band:int, diagonalExtent:float, heading:int = 0, dOffset:int = 0, rot:bool = False,rotAngle:int=0) -> None:
        """
        Produce a georeferenced image and save to disk.
        Provide a capture window in meters, and a 

        Parameters:
        -----------
            savePath         (str)   : The directory location to save the output file
            band             (int)   : What wavelength to perform this operation on, by index
            diagonalExtent   (float) : The extent by which the capture frame covers in meters
            heading          (int)   : The pygeo standard for bearing, used in image skew. In the case of North-up imagery, use default 0. 
            dOffset          (int)   : For non-aerial imagery, this offset serves to place the point of capture with subject, default 0.
            dOffsetBearing   (int)   : The direction for which to apply the offset to subject. [0:360]
        """
        # TODO:
        #   Assert:: hcube is not none, band index within valid range, savePath exists
        #   heading is not greater than 360, lat/long list is populated
        
        # Get the image
        if isinstance(self.hcube,np.ndarray):
            bandImg = self.hcube[:,:,band]
        else:
            self.hcube.load()
            bandImg = self.hcube.read_band(band)

        
        # Get the center point of image
        imgCenter = geopy.Point(self.latList,self.longList)
        
        if dOffset > 0:
            d = geopy.distance.distance(meters=dOffset)
            imgCenter = d.destination(point=imgCenter,bearing=heading)
        
        # Get the edges for aspect ratio
        d = geopy.distance.distance(meters= diagonalExtent/2)
        topLeftPoint = d.destination(point = imgCenter, bearing = -45).format_decimal()
        bottomRightPoint = d.destination(point = imgCenter, bearing = 135).format_decimal() #135

        # Compute the resolution 
        xs = []
        ys = []
        xs.append(topLeftPoint.split(',')[0])
        xs.append(bottomRightPoint.split(',')[0])

        ys.append(topLeftPoint.split(',')[1])
        ys.append(bottomRightPoint.split(',')[1])

        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)

        rows = bandImg.shape[0]
        cols = bandImg.shape[1]
        xres = (float(xmax) - float(xmin)) / float(rows)
        yres = (float(ymax) - float(ymin)) / float(cols)

        # Compute the pixel skew TODO: The math needs work
        radHeading = self._deg_to_radian(heading)
        y_skew = -1* np.sin(radHeading) * xres 
        x_skew = yres * np.sin(radHeading)

        # Make the transformation on the band, save as tif
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        geotransform =(float(ymin), xres, 0 , float(xmax), y_skew, xres)#(float(ymin), np.cos(radHeading)*xres, y_skew , float(xmax), x_skew, np.cos(radHeading)*yres)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(f'{savePath}\{self.imageName}_band{band}.tif',rows , cols, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(geotransform)
        outdata.SetProjection(sr.ExportToWkt())
        outdata.GetRasterBand(1).WriteArray(bandImg)
        outdata.FlushCache()
        outdata = None
        print(f"Image saved: {savePath}/{self.imageName}_band{band}.tif")

    def georef_pushbroom_band(self, savePath:str,band:int, diagonalExtent:float,heading:int = 0, dOffset:int = 0, rot:bool = False) -> None:
        """
        Produce a georeferenced image and save to disk.
        Quick and dirty f(x) to get the job done
        TODO merge functionality with georef_band for single, cleaner interface 

        Parameters:
        -----------
            savePath         (str)   : The directory location to save the output file
            band             (int)   : What wavelength to perform this operation on, by index
            diagonalExtent   (float) : The extent by which the capture frame covers in meters
            heading          (int)   : The pygeo standard for bearing, used in image skew. In the case of North-up imagery, use default 0. 
            dOffset          (int)   : For non-aerial imagery, this offset serves to place the point of capture with subject, default 0.
            dOffsetBearing   (int)   : The direction for which to apply the offset to subject. [0:360]
        """
        # TODO:
        #   Assert:: hcube is not none, band index within valid range, savePath exists
        #   heading is not greater than 360, lat/long list is populated

        # Get the image
        if isinstance(self.hcube,np.ndarray):
            bandImg = self.hcube[:,:,band]
        else:
            tifPath = self.save_tif(band=band,savePath=savePath,getPath=True,southUp=rot)
            bandImg = tifffile.imread(tifPath)
            os.remove(tifPath)

        
        # Get the center point of image
        imgTieStart = geopy.Point(self.latList[0],self.longList[0])
        imgTieEnd = geopy.Point(self.latList[0],self.longList[-1])
        
        if dOffset > 0:
            d = geopy.distance.distance(meters=dOffset)
            imgTieStart = d.destination(point=imgTieStart,bearing=heading)
            imgTieEnd = d.destination(point=imgTieEnd,bearing=heading)
        
        # Get the edges for aspect ratio
        d = geopy.distance.distance(meters= diagonalExtent/2)
        topLeftPoint = d.destination(point = imgTieStart, bearing = -45).format_decimal()
        bottomRightPoint = d.destination(point = imgTieEnd, bearing = 135).format_decimal() #135

        # Compute the resolution 
        xs = []
        ys = []
        xs.append(topLeftPoint.split(',')[0])
        xs.append(bottomRightPoint.split(',')[0])

        ys.append(topLeftPoint.split(',')[1])
        ys.append(bottomRightPoint.split(',')[1])

        xmin = min(xs)
        ymin = min(ys)
        xmax = max(xs)
        ymax = max(ys)
        # blurredPixelOffset = 300
        cols = 640
        rows = bandImg.shape[1]

        xres = (float(xmax) - float(xmin)) / float(cols)
        yres = (float(ymax) - float(ymin)) / float(rows)

        # Compute the pixel skew TODO: The math needs work
        # radHeading = self._deg_to_radian(heading)
        # y_skew = -1* np.sin(radHeading) * xres 
        # x_skew = yres * np.sin(radHeading)

        # Make the transformation on the band, save as tif
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        geotransform = (float(ymin), yres, 0, float(xmax), 0, xres)# (float(ymax), yres, 0, float(xmin), 0, xres)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(tifPath,rows , cols, 1, gdal.GDT_UInt16)#f'{savePath}\{self.imageName}.tif'
        outdata.SetGeoTransform(geotransform)
        outdata.SetProjection(sr.ExportToWkt())

        outdata.GetRasterBand(1).WriteArray(bandImg[:,:,0])
        outdata.FlushCache()
        outdata = None
        print(f"Image saved: {savePath}/{self.imageName}_band{band}.tif")
        
    def _deg_to_radian(self,deg:Union[int,float]) -> float:
        """
        Utility to convert degree inputs to radians for raster skew
        """
        return deg * math.pi/180

    def save_tif(self,band:int,savePath:str,getPath:bool = False,southUp:bool=False) -> Union[None,str]:
        # TODO: make sure band in range
        # TODO: make sure savePath exists
        """
        Saves a single band of the hypercube to disk as a tif file. 
        As used largely in the case of pushbroom data, a 90 deg rotation is applied by default.
        Handy for georeferencing images without loading all of hypercube to memory.

        Parameters:
        -----------
            band (int) : Wavelength to use from hcube
            savePath (str) : Location/directory of where to save the tif file
            getPath (bool) : if set to true, will return the absolute save path used. Default False.
            southUp (bool) : if set to true, will save the tif with inverse pixels. Default False.
        """
        if southUp:
            img = np.flip(np.rot90(np.flipud(self.hcube[:,:,band])),0)
        else:
            img = np.rot90(self.hcube[:,:,band])
        saveName = f"{join(savePath,self.imageName)}_band{band}.tif"
        tifffile.imwrite(f"{saveName}", img)
        if getPath:
            return saveName
        else:
            print(f"Saved band: {band} as tif file:\n {saveName}")

    def save_gps_csv(self,savePath:str) -> None:
        """
        Produces a csv of all gps points collected (single point in case of drone)

        Parameters:
        -----------
            savePath (str) : Location/directory of where to save the csv to
        """
        locations = {"latitude":self.latList, "longitude":self.longList}
        saveName = f"{join(savePath,self.imageName)}_NAV.csv"
        df = pd.DataFrame(locations)
        df.to_csv(saveName,index=False)
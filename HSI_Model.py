# from genericpath import isfile
from os.path import isfile,isdir, join
from spectral import *
import spectral.io.envi as envi
import numpy as np
import cv2
# sys.path.append("../Utilities")
import HDRprocess

def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))

class HSI_Model:
    """ An object designed to house: Hypercube, wavelengths used, RGB image, logical mask """


    def __init__(self, path_hcube:str, imgName:str, path_mask:str = None,dataset:str = None,norm_max:int = None) -> None:
        """ 
        Properties will contain all relevant meta-data for HSI image. 
        
        Parameters:
            path_hcube -- Absolute path to HDR file, assumes DAT file is within same directory
            imgName -- Camera given name of this HSI,
            path_mask -- Optional path to associated segmentation mask for this image,
            dataset -- Only options are 'berry' and 'tripod'. Defaults to tripod, optional parameter for rgb creation
            norm_max -- The max value given to the rgb composite of this image
        """
        
        self.imageName = None
        self.hcube = None
        self.wv = None
        self.rgb = None
        self.mask = None
        self.timeTaken = None
        self.gain = None
        
        
        print("loading hypercube...")
        envi_obj = envi.open(path_hcube)
        envi_obj = envi_obj.load()
        self.wv = envi_obj.bands.centers
        self.imageName = imgName
        if dataset is None:
            print(f"constructing composite image using wavelengths:{self.wv[73]}nm, {self.wv[14]}nm, {self.wv[6]}nm ")
            r = envi_obj[:,:,73] # red
            g = envi_obj[:,:,14] # green
            b = envi_obj[:,:,6]
            self.rgb = cv2.merge([r,g,b])
        elif dataset == "berry":
            print(f"constructing composite image using wavelengths:{self.wv[2]}nm, {self.wv[10]}nm, {self.wv[30]}nm ")
            # if norm_max is None: # default will be [0-1]
            r = norm(envi_obj[:,:,2])
            g = norm(envi_obj[:,:,10])
            b = norm(envi_obj[:,:,30])
            self.rgb = cv2.merge([r,g,b])

        if not norm_max is None:
            self.rgb = cv2.normalize(self.rgb,None,alpha=1,beta=norm_max,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)


        # Assign properties:
        self.hcube = np.array(envi_obj)
        if path_mask != None:
            self.load_mask(path_mask)
        self.timeTaken = HDRprocess.get_time(path_hcube)
        self.gain = HDRprocess.get_gain_array(path_hcube)
        print("HSI load complete...")

    def load_mask(self,path_mask):
        """ Called on initialization, if a path to multi-class mask is provided to constructor. """
        if isfile(path_mask):
            mask_multiclass = cv2.imread(path_mask)
            mask_multiclass = cv2.cvtColor(mask_multiclass, cv2.COLOR_BGR2RGB)
            self.mask = (mask_multiclass[:,:,2] == mask_multiclass[:,:,1]).astype(np.int8) * 255
    
    # def hist_equalize()
    def load_mask_from_model_output(self,maskObj):
        self.mask = maskObj

    def save_rgb(self,path_save_dir):
        """ Used to save the composite rgb image of HSI """
        cv2.imwrite(f"{join(path_save_dir,self.imageName)}.jpg",self.rgb)

    def save_mask(self,path_save_dir):
        cv2.imwrite(f"{join(path_save_dir,self.imageName)}.png",self.mask)

    def set_rgb_by_wv_index(self,r:int,g:int,b:int,norm_max:int =None) -> None:
        """Reset the auto-generated rgb manually by selecting the index of wavelengths, if no norm_max is specified then pixels will be decimals"""
        r = norm(self.hcube[:,:,r])
        g = norm(self.hcube[:,:,g])
        b = norm(self.hcube[:,:,b])
        self.rgb = cv2.merge([r,g,b])
        if not norm_max is None:
            self.rgb = self.rgb = cv2.normalize(self.rgb,None,alpha=1,beta=norm_max,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)


    def roi_equalization(image):
        """
        In progress... normalize image pixels with a region of interest
        """
        ROI = image[400:800,600:800]

    #  Calculate mean and STD
        mean, STD  = cv2.meanStdDev(ROI)

    #  Clip frame to lower and upper STD
        offset = 0.2
        clipped = np.clip(image, mean - offset*STD, mean + offset*STD).astype(np.uint8)

# Normalize to range
        result = cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)
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
        cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g, 0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


        cdf_m_r = np.ma.masked_equal(cdf_r, 0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
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
import numpy as np
import cv2

class ImageSegmentation:
    '''
    Class to be used to store the acquired images split in two channels and methods useful for cell identification and roi creation
    '''

    def __init__(self, imageRaw, half_side, min_cell_size):

        self.image = imageRaw
        _, self.dim_w, self.dim_h = np.shape(imageRaw)
        
        self.contour = []        # list of contours of the detected cells
        self.cx = []             # list of the x coordinates of the centroids of the detected cells
        self.cy = []             # list of the y coordinates of the centroids of the detected cells
        self.selected_cx = []    # list of the x coordinates of the centroids of the detected cells to be saved (not at the boundary of our acquired frame)
        self.selected_cy = []    # list of the y coordinates of the centroids of the detected cells to be saved (not at the boundary of our acquired frame)
        
        self.roi_half_side = half_side        # half dimension of the roi
        self.min_cell_size = min_cell_size    # minimum area that the object must have to be recognized as a cell

    def find_cell(self):
        """
        Determines if a region avove thresold is a cell, generates contours of the cells and their centroids cx and cy      
        """          
        image = np.sum(self.image,axis=0)
        level_min = np.amin(image)
        level_max = np.amax(image)
        img_thres = np.clip(image, level_min, level_max)
        self.image8bit = ((img_thres-level_min+1)/(level_max-level_min+1)*255).astype('uint8')

        # image8bit = (self.image[ch]/256).astype('uint8')
        # _ret,thresh_pre = cv2.threshold(self.image8bit,0,255,cv2.THRESH_OTSU)
        _ret,thresh_pre = cv2.threshold(self.image8bit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # thresh_pre = cv2.adaptiveThreshold(self.image8bit,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,211,2)
        # ret is the threshold that was used, thresh is the thresholded image.     
        kernel  = np.ones((3,3),np.uint8)

        thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_OPEN, kernel, iterations = 1)
        # morphological opening (remove noise,small holes)
        thresh = cv2.dilate(thresh,kernel,iterations = 1)
        contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cx = []
        cy = []            
        contour =[]
       
        for cnt in contours:
            
            M = cv2.moments(cnt)
            if M['m00'] >  int(self.min_cell_size):    # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)
                #extracts image center
                cx.append(int(M['m10']/M['m00']))
                cy.append(int(M['m01']/M['m00']))
                contour.append(cnt)
        
        self.cx = cx
        self.cy = cy 
        self.contour = contour  

    
    def draw_contours_on_image(self, image8bit):        
        """ Input: 
        img8bit: monochrome image, previously converted to 8bit
            Output:
        displayed_image: RGB image with annotations
        """  
        
        cx = self.cx
        cy = self.cy 
        roi_half_side = self.roi_half_side
        contour = self.contour
      
        displayed_image = cv2.cvtColor(image8bit,cv2.COLOR_GRAY2RGB)      
        
        for indx, _val in enumerate(cx):

            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
            w = h = roi_half_side*2

            displayed_image = cv2.drawContours(displayed_image, [contour[indx]], -1, (0,255,0), 0) 
            cv2.rectangle(displayed_image,(x,y),(x+w,y+h),(0,0,255),1)

        return displayed_image


    def roi_creation(self):

        image16bit = self.image
            
        cx = self.cx
        cy = self.cy 
        roi_half_side = self.roi_half_side
        
        l = image16bit.shape
        rois = []
        selected_cx = []
        selected_cy = []
        
        
        for indx, _val in enumerate(cx):
            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
            w = h = roi_half_side*2
            
            if x>0 and y>0 and x+w<l[2]-1 and y+h<l[1]-1:    # only rois far from edges are considered
                    detail = image16bit [:,y:y+w, x:x+h]
                    rois.append(detail)
                    selected_cx.append(cx[indx])
                    selected_cy.append(cy[indx])
                    
        self.selected_cx = selected_cx
        self.selected_cy = selected_cy
                    
        return rois


    def highlight_channel(self,displayed_image):
         cv2.rectangle(displayed_image,(0,0),(self.dim_h-1,self.dim_w-1),(255,255,0),3)
   

            
if __name__ == '__main__':
    
    import tifffile as tif
    
    image_cell = np.single(tif.imread('2021_0611_1751_488nm_Raw.tif'))
    h = ImageSegmentation(image_cell,128,50**2)
    h.find_cell()
    rois_cell = h.roi_creation()
    displayed_image =h.draw_contours_on_image(h.image8bit)
    cv2.imshow('image',displayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('show')
    #h.highlight_channel(displayed_image)
    #h.draw_contours_on_image
    
    
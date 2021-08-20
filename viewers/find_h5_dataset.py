"""Written by Andrea Bassi (Politecnico di Milano) 10 August 2018
to find the location of datasets in a h5 file.
"""

import h5py

def find_dataset(item):
        """Returns the DataSet within the HDF5 file and its shape. Found gives the number of dataset found"""
        [name,shape,found]= get_hdf5_item_structure(item, name=[], shape=[], found = 0)
        return (name,shape,found)
    
def get_hdf5_item_structure(g, name, shape, found) :
        """Extracts the dataset location (and its shape) and it is operated recursively in the h5 file subgroups  """
                
        if   isinstance(g,h5py.File) :
            found=found #this and others are unnecessary, but left for future modifications
               
        elif isinstance(g,h5py.Dataset) :
           
            found=found+1
            name.append(g.name)
            shape.append(g.shape)
               
        elif isinstance(g,h5py.Group) :
            found=found
            
        else :
            found=found
            print ('WARNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
            #sys.exit ( "EXECUTION IS TERMINATED" )
     
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
           
            for key,val in dict(g).items() :
                subg = val
                
                [name,shape,found]=get_hdf5_item_structure(subg,name,shape,found)
                 
        return (name,shape,found)    
        
            
"""The following is only to test the functions.
It will find a dataset and display it
"""     
if __name__ == "__main__" :
    
        import sys
        import pyqtgraph as pg
        import qtpy.QtCore
        from qtpy.QtWidgets import QApplication
        import numpy as np
        
        # this h5 file must contain a dataset composed by an array or an image
        file_name='C:\\Users\\Andrea Bassi\\OneDrive - Politecnico di Milano\\Data\\PROCHIP\\stackROI_h5\\test.h5'
        
        file = h5py.File(file_name, 'r') # open read-only
        [dataname,datashape,datafound] = find_dataset(file)    
        
        
        #show dataset location and shape        
        print('dataname:', dataname)
        print('datashape:', datashape)
        print('datafound', datafound)    
              
        #read data and plot
        #data = file[dataname[0]]
        #time = file[dataname[1]]
        #pg.plot(np.array( time, dtype = float), np.array( data, dtype = float), title="Acquired data from Thorlabs PD")        
        
        
        #read image and show
        stack = file[dataname[0]]
        pg.image(np.array(stack), title="Stack of images")        
               
               
        #keeps the window open running a QT application
        if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
            QApplication.exec_()
                          
        file.close()
        sys.exit ( "End of test")
"""Written by Andrea Bassi (Politecnico di Milano) 10 August 2018:
viewer compatible with Scopefoundry DataBrowser.
Finds an imaging dataset (3D stack of images) in the h5 file
If multiple datasets are found by the function find_h5dataset, the browser will show the first dataset found  
"""

from ScopeFoundry.data_browser import DataBrowser, DataBrowserView
from qtpy import QtWidgets
import h5py
import pyqtgraph as pg
import numpy as np
import os
from viewers.find_h5_dataset import find_dataset
import pdb

class ImageStackH5(DataBrowserView):

    name = 'stack_h5_view'
    
    def setup(self):
    
        self.settings.New('Dataset index', dtype=int, initial= 0, vmin = 0)
        self.settings.get_lq('Dataset index').add_listener(self.update_display)
        
        self.ui = QtWidgets.QWidget()
        self.ui.setLayout(QtWidgets.QVBoxLayout())
        self.ui.layout().addWidget(self.settings.New_UI(), stretch=0)
        self.info_label = QtWidgets.QLabel()
        self.ui.layout().addWidget(self.info_label, stretch=0)
        self.imview = pg.ImageView(view=pg.PlotItem())
        self.imview.show()
                
        self.ui.layout().addWidget(self.imview, stretch=0)
        
                
    def on_change_data_filename(self, fname):
        #pdb.set_trace()       
        try:
            self.search_h5_datasets(fname)
            self.file = f = h5py.File(fname)
            self.update_display()
                            
        except Exception as err:
            #self.imview.setImage(np.zeros((10,10,10)))
            self.databrowser.ui.statusbar.showMessage("failed to load %s:\n%s" %(fname, err))
            raise(err)
              
        
    def is_file_supported(self, fname):
        _, ext = os.path.splitext(fname)
        return ext.lower() in ['.h5']
    
      
    def update_display(self):
        #pdb.set_trace()   
        if hasattr(self,'imview'):
            self.set_stack()                                    
            self.imview.setImage(self.stack) 
                    
                                    
    def set_stack(self):
        stack_index = self.settings['Dataset index']
        
        if stack_index >= self.found:    
            stack_index = 0
            self.settings['Dataset index'] = stack_index    
        
        f= self.file
        self.stack = (np.array(f[self.dataname[stack_index]]))
        self.current_stack_index = stack_index
                    
               
    def search_h5_datasets(self, fname):
        f = h5py.File(fname)
        [self.dataname,self.shape,self.found]=find_dataset(f)
        
      
            
if __name__ == '__main__':
    import sys
    
    app = DataBrowser(sys.argv)
    app.load_view(ImageStackH5(app))
       
    sys.exit(app.exec_())

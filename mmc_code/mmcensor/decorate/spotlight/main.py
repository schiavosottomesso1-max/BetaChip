import importlib
import tkinter as tk
import cv2
import numpy as np
from mmcensor.decorate.decorator_utils import feature_selector
import math

class decorator:

    def initialize( self, known_classes ):
        self.known_classes = known_classes
        self.classes = []
        self.dark = 25
        self.strength = 3

    def decorate( self, img, boxes ):
        imgTmp = img.copy()
        factor = 2 * math.ceil( self.strength * min( img.shape[0:1] ) / 100 /2 ) + 1
        img = cv2.blur( img, (factor,factor), cv2.BORDER_DEFAULT )
        img = (img * (self.dark/100.)).clip(0,255).astype(np.uint8)
        for box in boxes:
            if self.known_classes[box[1]] in self.classes:
                img[box[3]:box[5],box[2]:box[4]] = imgTmp[box[3]:box[5],box[2]:box[4]]
                
        return img

    def export_settings( self ):
        return( { 'classes': self.classes, 'dark': self.dark, 'strength': self.strength } )

    def import_settings( self, settings ):
        self.classes = settings['classes']
        self.dark = settings['dark']
        self.strength = settings['strength']

    def short_name( self ):
        return 'window'

    def short_desc( self ):
        return '%d classes, dark %d, strength %d'%(len(self.classes),self.dark,self.strength )

    def populate_config_frame( self, frame ):
        #self.feature_selector = importlib.import_module('decorator_utils').feature_selector()
        self.dark_var = tk.IntVar()
        self.strength_var = tk.IntVar()

        tk.Label( frame, text="Dark (0-100)" ).grid(row=0,column=0)
        self.dark_entry = tk.Entry( frame, textvariable=self.dark_var )
        self.dark_entry.delete(0,tk.END)
        self.dark_entry.insert(0,str(self.dark))
        self.dark_entry.grid(row=0,column=1)
        
        tk.Label( frame, text="strength (1-50)").grid(row=1,column=0)
        self.strength_entry = tk.Entry( frame, textvariable=self.strength_var )
        self.strength_entry.delete(0,tk.END)
        self.strength_entry.insert(0,str(self.strength))
        self.strength_entry.grid(row=1,column=1)

        self.feature_selector = feature_selector()

        class_frame = tk.Frame(frame)
        self.feature_selector.populate_frame(class_frame, self.known_classes, self.classes)
        class_frame.grid(row=2,column=0,columnspan=2)

    def apply_config_from_config_frame( self ):
        self.classes = self.feature_selector.get_selected_classes()
        self.dark = self.dark_var.get()
        self.strength = self.strength_var.get()

    def destroy_config_frame( self ):
        return 0

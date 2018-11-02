#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:45:28 2018

@author: Musau
"""

from src import reluPlexPrinter
import tkinter as tk
import os
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(title="Select network model file",filetypes = (("nnet files","*.nnet"),("all files","*.*")))
    root.update() 
    #print(file_path)
    
    initDir=os.path.basename(os.path.normpath(file_path))
    outputdirectory=filedialog.askdirectory(initialdir=initDir,title="Select Output Directory")
    root.update()
    #print(outputdirectory)
    root.destroy()
    
    reluPlexPrinter.reluplexPrinter(file_path,outputdirectory)

    

main()

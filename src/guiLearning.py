#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:25:59 2018

@author: musaup
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os


class Window(tk.Frame):
    
    
    #Here you create a window class that inherits from the Frame Class. Frame is a class from the tkinter module.
    def __init__(self, master=None):
        #paramters that you want to send through to the Frame class
        tk.Frame.__init__(self, master)
        #reference to the master widget which is the tk window
        self.master=master
        #size of the window
        self.master.geometry("400x300")
        #With that we want then to run init_windo which doesn't yet exist
        self.init_window()
        
    #Creation of init Window
    def init_window(self):
        #changing the title of our master widget 
        self.master.title("NNMP: Neural Network Model Parser")
        #allow the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)
        #create an initial directory to search
        self.initial_dir=''
        
        #create a label for the combobox
        labelTop=tk.Label(self, text="Select the network model file")
        labelTop.grid(column=0,row=0)
        #create second label
        label2=tk.Label(self, text="Select output format")
        label2.grid(column=0, row=3)
        #create a Combo Box
        self.comboSelect= ttk.Combobox(self, values=["Select a model format...","Reluplex (.nnet)", "Sherlock(.txt)"])
        self.comboSelect.grid(column=0, row=2)
        self.comboSelect.current(0)
        
        #create output Combo Box
        self.comboOutput=ttk.Combobox(self, values=["Select an output format...","Matfile (.mat)", "Onnx (under dev)"])
        self.comboOutput.grid(column=0,row=4)
        self.comboOutput.current(0)
        
        #create a button to select the neural network model path that will edit the model entry box in the gui
        buttonModelEntry=tk.Button(self, text="Set Model Path", command=self.select_model_file)
        buttonModelEntry.grid(column=1,row=5)
        
        #create an entry so that the user can see which model or modes they hva selected
        self.modelFileEntryDefaultString=tk.StringVar(self, value='Path to model...')
        self.modelFileEntry=tk.Entry(self, state='disabled', textvariable=self.modelFileEntryDefaultString,disabledbackground= "white")
        self.modelFileEntry.grid(column=0, row=5)
        
        #create a button to select the model output path and edit the entry for the output path
        buttonModelOutput=tk.Button(self, text='Set Output Path',command=self.select_output_path)
        buttonModelOutput.grid(column=1, row=6)
        
        #create an entry so that the user can see which directory they have selected
        self.modelOutputDefaultString=tk.StringVar(self, value='Output path...')
        self.modelOutputPath=tk.Entry(self, state='disabled', textvariable=self.modelOutputDefaultString,disabledbackground= "white")
        self.modelOutputPath.grid(column=0, row=6)
        
    
        
        
        
        
        
        #create a button instance
        quitButton= tk.Button(self,text="Convert", command=self.client_exit)
        #placing the button on my window
        quitButton.grid(column=0,row=7)
    def select_model_file(self):
        filePath=filedialog.askopenfilename(title="Select network model file",filetypes = (("nnet files","*.txt"),("all files","*.*")))
        self.modelFileEntryDefaultString.set(filePath)
        self.initial_dir=os.path.basename(os.path.normpath(filePath))
    def select_output_path(self):
        outputdirectory=filedialog.askdirectory(initialdir=self.initial_dir,title="Select Output Directory")
        self.modelOutputdefaultString.set(outputdirectory)
    def client_exit(self):
        print(self.comboSelect.get(),"to", self.comboOutput.get())
        #self.master.destroy()
        
#root window created. Here that would be the only window but you can later have windows within windows
root=tk.Tk()


app=Window(root)
root.mainloop()







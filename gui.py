#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:45:28 2018

@author: Musau
"""

import tkinter as tk
from src import ToolGUI

def main():
    #root window created. Here that would be the only window but you can later have windows within windows
    root=tk.Tk()
    ToolGUI.Window(root)
    root.mainloop()
    
main()

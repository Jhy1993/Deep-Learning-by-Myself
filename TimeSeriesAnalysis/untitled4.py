# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:59:26 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
filepath = 'C:\\Users\\Jhy\\Desktop\\data.txt'
data = []
with open(filepath) as f:
    lines = f.readlines()
    for line in lines:
        temp = line.strip('').split(',')
        if temp[0] == '222180117136533000':    
            data.append(temp)

        
    

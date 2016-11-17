# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:28:24 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:

"""

class jhy():
    def __init__(self, **kwargs):
        self.a = kwargs.get('a', 1)
        self.b = kwargs.get('b')
        
    def jia(self):
        return self.a + self.b
if __name__ == '__main__':
    x = jhy(b=10)
    y = x.jia()
    print y * 2
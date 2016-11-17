# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 10:52:56 2016

@author: Jhy
"""

import pickle,pprint
#pkl=open('emotion.pickle','rb')
#data=pickle.load(pkl)
#保存
data1={'a':[1,2,3],
       'b':['s','ss']}
output=open('data.pkl','wb')
pickle.dump(data1,output)
#载入
pkl=open('data.pkl','rb')
data2=pickle.load(pkl)
pprint.pprint(data2)
pkl.close()
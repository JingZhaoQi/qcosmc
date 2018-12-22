# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:15:38 2017

@author: jingzhao
"""

from qicosmc.plotfile import Qplot

rdic=[]
rdic.append(['wcdm_OHD','OHD'])
rdic.append(['wcdm_BAO','BAO'])
rdic.append(['wcdm_CMB','CMB'])
rdic.append(['wcdm_SN','SN'])
rdic.append(['wcdm_all','all'])
Qplot(rdic,plot1D=[0,2],plot2D=[1,1,2],plot3D=[0,0],results=0)
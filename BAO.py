# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:28:27 2019

@author: qijingzhao
"""
import numpy as np
import matplotlib.pyplot as plt
import os

dataDir=os.path.dirname(os.path.abspath(__file__))+'/data'



class BAO_likelihood(object):
    def __init__(self,data_set):
        self.data_set = data_set
    
    
    
    def __read(self,dataname):
        z,bao=np.loadtxt(dataname,unpack=True)
        cov=np.loadtxt(dataname[0:-4]+'_invc.txt')
        return z,bao,np.reshape(cov,(len(z), len(z)))
    
    def read_Data(self):
        self.z_dr12,self.dr12,self.dr12_cov = self.__read(dataDir+'/DR12.txt')
        self.rd_f = 150.78
        self.z_6dFGS= 0.106
        self.dFGS = 3.047
        self.dFGS_s = 0.137
        self.z_MGS = 0.15
        self.MGS = 4.48
        self.MGS_s = 0.168
        self.xx=np.zeros(len(self.z_dr12))
#    if 'DR12' in self.data_set:
#        z,bao,cov=self.__read(self.dataDir+'/DR12.txt')
    
        
    
    def chiBAO(self,model):
        self.xx[0:5:2]=model.DM_rd(self.z_dr12[0:5:2],self.rd_f)-self.dr12[0:5:2]
        self.xx[1:6:2]=model.H_rd(self.z_dr12[1:6:2],self.rd_f)-self.dr12[1:6:2]
        chi=np.dot(self.xx,np.dot(self.dr12_cov,self.xx))
        chi+=(1/model.rs_over_Dv(self.z_MGS)-4.48)**2/0.168**2
        chi+=(model.rs_over_Dv(self.z_6dFGS)-0.336)**2/0.015**2
        return chi
        
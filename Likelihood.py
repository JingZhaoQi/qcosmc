# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:29:07 2018

@author: qijingzhao
"""
import numpy as np
import scipy.special as S
import scipy.constants as sc
import os
import pandas as pd
from .MCMC import MCMC_class
#print( os.getcwd())
#print(os.path.dirname(os.path.abspath(__file__)))
dataDir=os.path.dirname(os.path.abspath(__file__))+'/data/'
class likelihood(object):
    def __init__(self,cosModel,param):
        self.cosModel = cosModel
        self.param = param
        self.pn=len(param)
        self.infor='The data used are: '
        self.data_num=0
        self.SN2=lambda x:0
        self.JLA2=lambda x:0
        self.OHD2=lambda x:0
        
#==============================================================================================    
    @property
    def _read_SN(self):
        '''
        Data from 'The Astrophysical Journal, 2012, 746(1):85'
        '''
        (self._zsn,self._u_obs,self._u_err)=np.loadtxt(dataDir+"sn.txt", unpack='True')
        return len(self._zsn)

    def chiSN(self,theta):
        ld = self.cosModel(theta[0:self.pn]).lum_dis_z
        mu_th = 5.0*np.log10(ld(self._zsn))
        A=np.sum(((mu_th-self._u_obs)/self._u_err)**2)
        B=np.sum((mu_th-self._u_obs)/self._u_err**2)
        C=np.sum(1.0/self._u_err**2)
        kaf=A-B**2/C
        return kaf
    
    @property
    def addSN(self):
        snn=self._read_SN
        self.SN2=self.chiSN
        self.infor=self.infor+'SN(%s)+'%snn
        self.data_num=self.data_num+snn
#==================================JLA============================================================
    def __covRead(self, file_name):
        tmp = np.fromfile(dataDir + file_name, sep=" ") 
        n = int(tmp[0])
        cov = tmp[1:]
        cov = np.reshape(cov, (n,n))
        return cov
    
    @property
    def _read_JLA(self):
        jla=pd.read_csv(dataDir+'jla_lcparams.txt',sep='\s+',index_col=0)
        jn=len(jla)
        self._zcmb=jla['zcmb'].values
        self._zhel=jla['zhel'].values
        self._mb=jla['mb'].values
        self._dmb=jla['dmb'].values
        self._x1=jla['x1'].values
        self._dx1=jla['dx1'].values
        self._color=jla['color'].values
        self._dcolor=jla['dcolor'].values
        self._3rdvar=jla['3rdvar'].values
        self._cov_m_s=jla['cov_m_s'].values
        self._cov_m_c=jla['cov_m_c'].values
        self._cov_s_c=jla['cov_s_c'].values
        self._v0 = self.__covRead('jla_v0_covmatrix.dat')
        self._va = self.__covRead('jla_va_covmatrix.dat')
        self._vb = self.__covRead('jla_vb_covmatrix.dat')
        self._v0a = self.__covRead('jla_v0a_covmatrix.dat')
        self._v0b = self.__covRead('jla_v0b_covmatrix.dat')
        self._vab = self.__covRead('jla_vab_covmatrix.dat')
        self._prob=np.zeros(len(self._zcmb))
        self._prob[np.where(self._3rdvar>10)]=1.0
        self.jlap_n=len(self.param)
        return jn
    
    def chiJLA(self,theta):
        # JLA nuiance parameters 
        # ==========================
        # p[0]:alpha
        # p[1]:beta
        # p[2]:M
        # p[3]:DeltaM
        p=theta[self.jlap_n:self.jlap_n+4]
        cov_stat_sys=self._v0+p[0]**2*self._va+p[1]**2*self._vb+2*p[0]*self._v0a-2*p[1]*self._v0b-2*p[0]*p[1]*self._vab
        Dstat=self._dmb**2+(p[0]*self._dx1)**2+(p[1]*self._dcolor)**2+2*p[0]*self._cov_m_s-2.0*p[1]*self._cov_m_c-2.0*p[0]*p[1]*self._cov_s_c
        covMatrix=np.diag(Dstat)+cov_stat_sys
        mu_sn=self._mb+p[0]*self._x1-p[1]*self._color-p[2]-p[3]*self._prob
        mu_th=5.0*np.log10( (1.0+self._zhel)* self.cosModel(theta[0:self.pn]).co_dis_z(self._zcmb))+25.0
        res=mu_sn-mu_th
        residuals = np.dot(res,np.dot(np.linalg.inv(covMatrix),res))
        return residuals

    @property
    def addJLA(self):
        jla_param=[['\\alpha_{JLA}',0.135,0.1,0.2],
                   ['\\beta_{JLA}',3.1,2.8,3.4],
                   ['M_B^1',-19.00,-19.2,-18.9],
                   ['\\Delta M',-0.07,-0.2,0]]
        jlan=self._read_JLA
        self.param=self.param+jla_param
        self.JLA2=self.chiJLA
        self.infor=self.infor+'JLA(%s)+'%jlan
        self.data_num=self.data_num+jlan
#=============================================================================================
#=============================OHD likelihood=====================================   
    @property
    def _read_OHD(self):
        ''' Data from ' The Astrophysical Journal, Volume 838, Number 2
                        DOI:10.3847/1538-4357/aa674b
                        arXiv:1611.00904
                        An Improved Method to Measure the Cosmic Curvature'
        '''
        (self._zhz,self._Hz_obs,self._Hz_err)=np.loadtxt(dataDir+"OHD.txt",unpack='True')
        return len(self._zhz)
  
    def chiOHD(self,theta):
#        h_obs=(self.Hz_obs/H0)
#        h_err=(np.sqrt(self.Hz_err**2/H0**2+(self.Hz_obs**2/H0**4)*H0_s**2))
        kaf=np.sum((self.cosModel(theta[0:self.pn]).hubz(self._zhz)*self.cosModel(theta[0:self.pn]).h*1e2-self._Hz_obs)**2/self._Hz_err**2)
        return kaf
    
    @property
    def addOHD(self):
        ohdn=self._read_OHD
        self.OHD2=self.chiOHD
        self.infor=self.infor+'OHD(%s)+'%ohdn
        self.data_num=self.data_num+ohdn
            
#============================================================================================
    def chi2(self,theta):
        return self.JLA2(theta)+self.SN2(theta)+self.OHD2(theta)
    
    def MCMC(self,chain_name):
        print ('\n'+'='*60)
        print(self.infor[:-1])
        MC=MCMC_class(self.param,self.chi2,chain_name,self.data_num)
        MC.MCMC()
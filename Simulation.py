# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 10:53:16 2017

@author: jingzhao
"""
import numpy as np
from .tools import qplt,simp_err,mu_to_Dl,err_ts
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import splrep,splev
from .cos_models import LCDM
import pandas as pd
from scipy.constants import c,arcsec,parsec
import scipy.stats as stats
#from astroML.density_estimation import FunctionDistribution
Mpc=parsec*1e6
c0=c
'''
Generate the simulated data, including QSO, Strong lens (LSST .etc), SNIa
'''
class gen_QSO(object):
    def __init__(self,number,L=11.03,L_err=0.0,theta_err=0.03,Omegam=0.3,h=0.7,random_err=True,endpoint=False):
        self.num=number
        self.L=np.float64(L)
        self.L_s = np.float64(L_err)
        self.theta_err=np.float64(theta_err)
        self.h = np.float64(h)
        self.ll = LCDM(Omegam,h)
        self.random_err = random_err
        self.endpoint = endpoint
        
        
    #========================Luminosity function to Number of QSO========================================
    def Luminosity_function(self,Mg,z):
        h=self.h
        bh,bl,phis,Mgs0,k1,k2=[3.31,1.45,1.83e-6*(h/0.7)**3,-21.61+5*np.log(h/0.7),1.39,-0.29]
        Mgs=Mgs0-2.5*(k1*z+k2*z**2)
        fm=10**(0.4*(1-bh)*(Mg-Mgs))+10**(0.4*(1-bl)*(Mg-Mgs))
        return phis/fm
    
    def qz(self):
        number=self.num
        Mgi=[18,23]
        bins_with=0.2
        zz=np.arange(0.5,6+bins_with,bins_with)
        N=len(zz)
        Nqso=np.zeros(N-1)
        bins=np.zeros([N-1,2])
        addn=0
        
        for i in range(N-1):
            bins[i:]=[zz[i],zz[i+1]]
            Nqso[i]=integrate.nquad(self.Luminosity_function,[[Mgi[0],Mgi[1]],[zz[i],zz[i+1]]])[0]
        
        ratio=Nqso/np.sum(Nqso)
        true_n=number-1
        while true_n<number:
            fb=list(map(int,map(round,(number+addn)*ratio)))
            true_n=sum(fb)
            addn=addn+1
        zzn=[]
        for i in range(len(fb)):
            zn=np.random.uniform(bins[i,0],bins[i,1],fb[i])
            zzn=np.append(zzn,zn)
        if self.endpoint:
            zzn=np.append(zzn,[0.5,6])
        ans=np.random.choice(zzn,number,replace=False)
    #    width=(zz[1]-zz[0])/2
    #    plt.bar(zz[0:N-1]+width, Nqso, alpha = .5, color = 'g',width = width)
    #    plt.xlabel('$z$')
    #    plt.ylabel('Number of QSO')
        return np.sort(ans)
    
    
    #========================Simulate the QSO================================
    def _theta(self,L0,z):
        DA=self.ll.ang_dis_z(z)
        return L0*1e-6/DA*180*3600/np.pi*1e3
    
    def _DDA(self,L0,thetaq):
        DA=L0/(thetaq*1e-3/(180/np.pi*3600))/1e6
        return DA
    
    def theta(self):
        qzz=self.qz()
        theta_th=self._theta(self.L,self.qz())
        theta_s=theta_th*self.theta_err
        if self.random_err:
            theta_obs=np.random.normal(theta_th,theta_s)
        else:
            theta_obs=stats.truncnorm(-1,1,loc=theta_th,scale=theta_s).rvs()
        return qzz,theta_obs,theta_s
    
    def DA(self):
        qzz,the_th,the_s=self.theta()
        Da=self.L*1e-3/(the_th*arcsec)
        sig_Da=Da*np.sqrt((self.L_s)**2+self.theta_err**2)
        return qzz,Da,sig_Da
    
    def DA_th(self):
        z=self.qz()
        DA=self.ll.ang_dis_z(z)
        return DA
    
    def dp(self):
        DH=self.ll.D_H()
        z,da,da_s=self.DA()
        dp=da*(1+z)/DH
        dp_s=da_s*(1+z)/DH
        return z,dp,dp_s
            
    def qz_hist(self,bins=50,*args, **kwargs):
        plt.figure()
        plt.hist(self.qz(),bins=bins,normed=True,color='g',alpha=0.5,edgecolor='k')
        plt.yticks([])
        plt.xlim(0,6)
        plt.xlabel('Redshift')
        qplt(*args, **kwargs)

    def plt_DA(self,*args, **kwargs):
        qzz,DAq,DA_s=self.DA()
        plt.figure()
        plt.errorbar(qzz,DAq,DA_s,fmt='_')
        plt.xlabel('$z$')
        plt.ylabel('$D_A(z) ~(\mathrm{Mpc})$')
        qplt(*args, **kwargs)

    def save_DA(self,path_file_name):
        st=['#z','DA','DA_err']
        data=self.DA()
        dc=dict(zip(st,data))
        df=pd.DataFrame(dc)
        if path_file_name[-3:]=='lsx':
            df.to_excel(path_file_name,index=False)
        elif path_file_name[-3:]=='txt':
            df.to_csv(path_file_name,index=False,sep=' ')
        else:
            df.to_csv(path_file_name,index=False,sep=' ')
    
    def save_theta(self,path_file_name):
        st=['#z','theta','theta_err']
        data=self.theta()
        dc=dict(zip(st,data))
        df=pd.DataFrame(dc)
        if path_file_name[-3:]=='lsx':
            df.to_excel(path_file_name,index=False)
        elif path_file_name[-3:]=='txt':
            df.to_csv(path_file_name,index=False,sep=' ')
        else:
            df.to_csv(path_file_name,index=False,sep=' ')

class gen_SGL(object):
    def __init__(self,filename,thetaE_sig=0.01,sig_vv_sig=0.05,TD_sig=0.05,
                 middle_mass=True,Omegam=0.3,h=0.7,random=True,Accuracy=0.2,
                 standardisable=True):
        self.filename = filename
        self.ll = LCDM(Omegam,h)
        self.sgll=np.loadtxt(self.filename)
        self.middle_mass = middle_mass
        self.thetaE_sig = thetaE_sig
        self.sig_vv_sig = sig_vv_sig
        self.TD_s = TD_sig
        self.random = random
        self.Accuracy = Accuracy
        self.standardisable = standardisable
        self.index = []
        if middle_mass:
            sgn= [i for i in range(len(self.sgll[:,0])) if 100<=self.sgll[i][3]<=300]
            self.sgl=self.sgll[sgn]
        else:
            self.sgl=self.sgll
        
        
    def list_z(self,mn,**kwargs):
        thetaE=self.sgl[:,2]
        xs=self.sgl[:,9]
        ys=self.sgl[:,10]
        mX_i=self.sgl[:,16]
        beta=np.sqrt(xs**2+ys**2)
        y=beta/thetaE
        mu=(y**2+2.0)/y/np.sqrt(y**2+4.0)
        mX_obs=mX_i-2.5*np.log10(mu)
#        num_td=len(beta)
        t_err=0.0003
#        sgn= [i for i in range(len(beta)) if 2*beta[i]>=t_err]
        if self.standardisable:
            sgn= [i for i in range(len(beta)) if beta[i]>=t_err and mX_obs[i]<=22.15 and thetaE[i]>=0.9]
        else:
            sgn= [i for i in range(len(beta)) if beta[i]>=t_err and  thetaE[i]<=0.9]
        if 'zsrange' in kwargs:
            [zmin,zmax]=kwargs['zsrange']
            sgn= [i for i in sgn if zmin<=self.sgl[i][1]<=zmax]
        if 'zlrange' in kwargs:
            [zmin,zmax]=kwargs['zlrange']
            sgn= [i for i in sgn if zmin<=self.sgl[i][0]<=zmax]        
        sgln=np.random.choice(sgn,mn,replace=False)
        self.index=sgln
        return mX_obs[sgln]


    
    def data_zs(self,qzs,err=5e-3,pair_all=False):
        num=len(qzs)
        sgln=[]
        qn=[]

        if pair_all:
            for i in range(num):
                c1=[]
                errp=err
                while not list(c1):
                    c1=np.where(abs(qzs[i]-self.sgl[:,1])<=errp)[0]
                    errp=errp+err/5.0
                print('[%s] Matching redshift error is %s'%(i,errp))
                sgln.append(np.random.choice(c1))
                qn.append(i)
            self.index=sgln
        else:
            for i in range(num):
                c1=np.where(abs(qzs[i]-self.sgl[:,1])<=err)
                if list(c1[0]):
                    sgln.append(np.random.choice(c1[0]))
                    qn.append(i)
            self.index=sgln
        return qn
    
    def data_zlzs(self,dnum,qz,qDA,qDA_sig,err=5e-3):
        sgln=[]
        zl=self.sgl[:,0]
        zs=self.sgl[:,1]
        DAl=DAs=DAl_sig=DAs_sig=[]
        xs=self.sgl[:,9]
        ys=self.sgl[:,10]
        beta=np.sqrt(xs**2+ys**2)
#        num_td=len(beta)
        t_err=0.0003
        td= [i for i in range(len(beta)) if 2*beta[i]>=t_err]
        nn=np.random.choice(td,len(td),replace=False)
        var=0
        for i in nn:
            if var==dnum:
                break
            nzl=np.where(abs(zl[i]-qz)<=err)[0]
            nzs=np.where(abs(zs[i]-qz)<=err)[0]
            if len(nzl)>=1: nzl=np.random.choice(nzl)
            if len(nzs)>=1: nzs=np.random.choice(nzs)
            if nzl and nzs:
                DAl=np.append(DAl,qDA[nzl])
                DAl_sig=np.append(DAl_sig,qDA_sig[nzl])
                DAs=np.append(DAs,qDA[nzs])
                DAs_sig=np.append(DAs_sig,qDA_sig[nzs])
                sgln.append(i)
                var=var+1
        self.index=sgln
        return DAl,DAl_sig,DAs,DAs_sig
    
    def data_zlzs_gp(self,qz,qDA,qDA_sig):
        num_sgl=len(self.sgl[:,0])
        sgn= [i for i in range(num_sgl) if qz.min()<=self.sgl[i][1]<=qz.max() and
              qz.min()<=self.sgl[i][0]<=qz.max()]
        self.index=sgn
        tck_da=splrep(qz,qDA)
        tck_s=splrep(qz,qDA_sig)
        DAl=splev(self.sgl[sgn,0],tck_da)
        DAl_sig=splev(self.sgl[sgn,0],tck_s)
        DAs=splev(self.sgl[sgn,1],tck_da)
        DAs_sig=splev(self.sgl[sgn,1],tck_s)
        return DAl,DAl_sig,DAs,DAs_sig
    
    def data_timedl(self,dnum,qz,qDA,qDA_sig):
#        sgln=len(self.sgl[:,0])
        xs=self.sgl[:,9]
        ys=self.sgl[:,10]
        beta=np.sqrt(xs**2+ys**2)
        num_td=len(beta)
        err=1e-5
        while num_td>dnum+50 :
            td= [i for i in range(len(beta)) if 2*beta[i]>=err]
            num_td=len(td)
            err=err+0.001
        nn=np.random.choice(td,dnum,replace=False)
        sgn= [i for i in nn if qz.min()<=self.sgl[i][1]<=qz.max() and
              qz.min()<=self.sgl[i][0]<=qz.max()]
        self.index=sgn
        tck_da=splrep(qz,qDA)
        tck_s=splrep(qz,qDA_sig)
        DAl=splev(self.sgl[sgn,0],tck_da)
        DAl_sig=splev(self.sgl[sgn,0],tck_s)
        DAs=splev(self.sgl[sgn,1],tck_da)
        DAs_sig=splev(self.sgl[sgn,1],tck_s)
        return DAl,DAl_sig,DAs,DAs_sig
    
    def timeDL(self):
        '''
        return ::
        thetaE,tehtaA,thetaB : arcseconds
        time_delay,time_delay_sig : day
        '''
        sgln=self.index
        zl=self.sgl[sgln,0]
        zs=self.sgl[sgln,1]
        thetaE=self.sgl[sgln,2]
        sig_vv=self.sgl[sgln,3]
        xs=self.sgl[sgln,9]
        ys=self.sgl[sgln,10]
        beta=np.sqrt(xs**2+ys**2)
        
        Dl=np.vectorize(self.ll.ang_dis_z)(zl)
        Ds=np.vectorize(self.ll.ang_dis_z)(zs)
        Dls=np.vectorize(self.ll.ang_dis_z2)(zl,zs)
        Dt=Dl*Ds/Dls
        
        thetaA=(thetaE+beta)*arcsec
        thetaB=(thetaE-beta)*arcsec
        
        TD_t=(1.0+zl)/2.0/c0*Dt*(thetaA**2-thetaB**2)*Mpc/24./3600.
        TD_sig=np.abs(self.TD_s*TD_t)
        return zl,zs,thetaE,sig_vv,thetaA/arcsec,thetaB/arcsec,TD_t,TD_sig
    
    def magnifications(self):
        sgln=self.index
        thetaE=self.sgl[sgln,2]
        xs=self.sgl[sgln,9]
        ys=self.sgl[sgln,10]
        beta=np.sqrt(xs**2+ys**2)
        y=beta/thetaE
        def mu_plus(gamma):
            return 1./(1.-(3.-gamma)*np.power(1./(1.+y),gamma-1.))
        def mu_sum(gamma):
            mu_p=mu_plus(gamma)
            mu_m=1./(1.-(3.-gamma)*np.power(1./(1.-y),gamma-1.))
            return np.abs(mu_p)+np.abs(mu_m)
        if self.standardisable:
            mu,mu_sig=err_ts(mu_plus,2.0,0.02)
        else:
            mu,mu_sig=err_ts(mu_sum,2.0,0.02)
        return mu,mu_sig
    
    def dl(self):
        zl,zs,thE,sig_vv,thA,thB,Tt,Tsig=self.timeDL()
        thetaE,thetaA,thetaB=thE*arcsec,thA*arcsec,thB*arcsec
        TD_t,TD_sig=Tt*24.0*3600,Tsig*24.0*3600
        DH=self.ll.D_H()
        def solve_dl(dt,thE,sig_vv,thA,thB,zl):
            R_obs=2*c*dt/(1+zl)/(thA**2-thB**2)/Mpc
            D_obs=thE*c0**2/(4*np.pi*(sig_vv*1e3)**2)
            dl=R_obs*D_obs/DH*(1+zl)
            return dl
        if self.random:
            td_th=stats.truncnorm(-self.Accuracy,self.Accuracy,loc=TD_t, scale=TD_sig).rvs()
#            td_sig=np.random.normal(0.0,TD_sig)
            td_sig = TD_sig
            tE = stats.truncnorm(-self.Accuracy,self.Accuracy,thetaE,thetaE*self.thetaE_sig).rvs()
#            tE_sig = np.random.normal(0.0,thetaE*self.thetaE_sig)
            tE_sig = thetaE*self.thetaE_sig
            sig_th = stats.truncnorm(-self.Accuracy,self.Accuracy,sig_vv,sig_vv*self.sig_vv_sig).rvs()
#            sig_sig = np.random.normal(0.0,sig_vv*self.sig_vv_sig)
            sig_sig = sig_vv*self.sig_vv_sig
        else:
            td_th = TD_t
            td_sig = TD_sig
            tE = thetaE
            tE_sig = thetaE*self.thetaE_sig
            sig_th = sig_vv
            sig_sig = sig_vv*self.sig_vv_sig
        dl,dl_s=simp_err(solve_dl,[td_th,tE,sig_th,thetaA,thetaB,zl],
                         [td_sig,tE_sig,sig_sig,thetaA*0,thetaB*0,zl*0])
        return dl,dl_s
    
    def D(self):
        sgln=self.index
        thetaE=self.sgl[sgln,2]*arcsec
        sig_vv=self.sgl[sgln,3]
        def D_obs(thE,sigv):
            return thE*c0**2/(4*np.pi*(sigv*1e3)**2)
        if self.random:
            tE = stats.truncnorm(-self.Accuracy,self.Accuracy,thetaE,thetaE*self.thetaE_sig).rvs()
#            tE_sig = np.random.normal(0.0,thetaE*self.thetaE_sig)
            tE_sig = thetaE*self.thetaE_sig
            sig_th = stats.truncnorm(-self.Accuracy,self.Accuracy,sig_vv,sig_vv*self.sig_vv_sig).rvs()
#            sig_sig = np.random.normal(0.0,sig_vv*self.sig_vv_sig)
            sig_sig = sig_vv*self.sig_vv_sig
        else:
            tE = thetaE
            tE_sig = thetaE*self.thetaE_sig
            sig_th = sig_vv
            sig_sig = sig_vv*self.sig_vv_sig
        Dth,D_s=simp_err(D_obs,[tE,sig_th],[tE_sig,sig_sig])
        return Dth,D_s
    
    def return_SGL(self):
        sgln=self.index
        zl=self.sgl[sgln,0]
        zs=self.sgl[sgln,1]
        thetaE=self.sgl[sgln,2]
        sig_vv=self.sgl[sgln,3]
        return zl,zs,thetaE,sig_vv
#        return self.sgl[sgln,:]
    
    def dls(self,ds,ds_s):
        zl,zs,thetaE,sig_vv=self.return_SGL()
        thE=thetaE*arcsec
        def dls_f(thE,sig_vv,ds):
            D_obs=thE*c0**2/(4*np.pi*(sig_vv*1e3)**2)
            dlss=D_obs*ds
            return dlss
        if self.random:
            tE = stats.truncnorm(-self.Accuracy,self.Accuracy,thE,thE*self.thetaE_sig).rvs()
#            tE_sig = np.random.normal(0.0,thE*self.thetaE_sig)
            tE_sig = thE*self.thetaE_sig
            sig_th = stats.truncnorm(-self.Accuracy,self.Accuracy,sig_vv,sig_vv*self.sig_vv_sig).rvs()
#            sig_sig = np.random.normal(0.0,sig_vv*self.sig_vv_sig)
            sig_sig = sig_vv*self.sig_vv_sig
        else:
            tE = thE
            tE_sig = thE*self.thetaE_sig
            sig_th = sig_vv
            sig_sig = sig_vv*self.sig_vv_sig
        dlss,dls_s=simp_err(dls_f,[tE,sig_th,ds],
                           [tE_sig,sig_sig,ds_s])
        return dlss,dls_s

class gen_SNIa(object):
    '''
    WFIRST http://wfirst.gsfc.nasa.gov/
    Eur. Phys. J. C (2017) 77:434
    DOI 10.1140/epjc/s10052-017-5005-4
    '''
    def __init__(self,Omegam=0.3,h=0.7,additional_mu_err=0.0):
        self.ll=LCDM(Omegam,h)
        self.additional_mu_err = additional_mu_err
    
    def mu(self,z):
        def rn(z):
            z1=np.arange(0.1,1.8,0.1)
            n=[i for i in range(len(z1)) if z1[i]<=z]
            N=[69,208,402,223,327,136,136,136,136,136,136,136,136,136,136,136]
            return np.float64(N[n[-1]])
        def mu_err(z):
            s_meas=0.08
            s_int=0.09+self.additional_mu_err
            s_lens=0.07*z
#            s_stat2=(s_meas**2+s_int**2+s_lens**2)/np.vectorize(rn)(z)
            s_stat2=(s_meas**2+s_int**2+s_lens**2)
            s_sys=0.01*(1+z)/1.8
            s_tot=np.sqrt(s_stat2+s_sys**2)
            return s_tot
        Dl=np.vectorize(self.ll.lum_dis_z)
        mu_th=5.0*np.log10(Dl(z))+25.
#        return mu_th,mu_err(z)
        return mu_th,mu_err(z)
    
    def sn_num(self,number):
        N=[69,208,402,223,327,136,136,136,136,136,136,136,136,136,136,136]
        ratio=np.asarray(N)/float(np.sum(N))
        true_n=number-1
        addn=0
        bins=np.arange(0.1,1.8,0.1)
        while true_n<number:
            fb=map(int,map(round,(number+addn)*ratio))
            true_n=sum(fb)
            addn=addn+1
        zzn=[]
        for i in range(len(bins)-1):
            zn=np.random.uniform(bins[i],bins[i+1],fb[i])
#            print(zn,i)
            zzn=np.append(zzn,zn)
#        if self.endpoint:
#            zzn=np.append(zzn,[0.5,6])
        redshift=np.random.choice(zzn,number,replace=False)
    #    width=(zz[1]-zz[0])/2
    #    plt.bar(zz[0:N-1]+width, Nqso, alpha = .5, color = 'g',width = width)
    #    plt.xlabel('$z$')
    #    plt.ylabel('Number of QSO')
        return np.sort(redshift)
    
    def DL(self,z):
        mu_th,mu_s=self.mu(z)
        DLs,DL_s=mu_to_Dl(mu_th,mu_s)
        return DLs,DL_s
    
    def dp(self,z):
        DH=self.ll.D_H()
        DLs,DLs_s=self.DL(z)
        ds=DLs/(1+z)/DH
        ds_s=DLs_s/(1+z)/DH
        return ds, ds_s
    
    def dp_lens(self,z,mud,mud_err=0.05):
        '''
        mud is magnifications
        '''
        mu_x,mu_s=self.mu(z)
        dp=np.power(10.0,mu_x/5.0-5.0)/(1+z)/self.ll.D_H()
        dps=np.sqrt((dp*mud_err/2.0)**2+(np.log(10.0)/5.0*dp*mu_s)**2)
        return dp,dps
#        mobs,m_s=mu_x-2.5*np.log10(mud),mu_s-2.5*np.log10(mud)
#        def dps(zs,muobs,mD):
#            return np.power(10.0,(muobs+2.5*np.log10(mD))/5.0-5.0)/(1+zs)
#        dp,dp_s=simp_err(dps,[z,mobs,mud],[z*0.0,m_s,mud*mud_err])
#        return dp/self.ll.D_H(), dp_s/self.ll.D_H()

    def dp_lens2(self,z,mud,mud_err):
        mu_x,mu_s=self.mu(z)
        mobs=mu_x-2.5*np.log10(mud)
        def dps(zs,muobs,mD):
            return np.power(10.0,(muobs+2.5*np.log10(mD))/5.0-5.0)/(1+zs)
        dp,dp_s=simp_err(dps,[z,mobs,mud],[z*0.0,mu_s,mud_err])
        return dp/self.ll.D_H(), dp_s/self.ll.D_H()
class gen_drift(object):
    
    def __init__(self,year,Omegam=0.3,h=0.7):
        self.ll=LCDM(Omegam,h)
        self.H0 = h*1e2
        self.year = np.float64(year)
    
    def drift(self,z):
        mpc=Mpc*1e2
        c=c0*1e2
        year=self.year*365.0*24.0*60.*60.
        H0=self.H0*1e5/mpc
        return c*H0*year*(1-self.ll.hubz(z)/(1+z))
    
    def drift_s(self,z):
        def drift_err(z):
            if 2.<=z and z<=4.:
                q=-1.7
            elif z>4.0:
                q=-0.9
            else:
                q=0
                
            SN=3000.
            N=30.
            dd=1.35*(2370.0/SN)/np.sqrt(N/30.)*((1.+z)/5.)**q
            return dd
        return np.vectorize(drift_err)(z)
    
    def normal_data(self):
        z_d=np.asarray([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        return z_d,self.drift(z_d),self.drift_s(z_d)
    
    def dtoh(self,dv,z):
        mpc=Mpc*1e2
        c=c0*1e2
        year=self.year*365.0*24.0*60.*60.
        H0=self.H0*1e5/mpc
        return (1-dv/c/H0/year)*(1+z)
    
    def Ez(self,z):
        Ezz,Ez_s=simp_err(self.dtoh,[self.drift(z),z],[self.drift_s(z),z*0])
        return Ezz,Ez_s
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:23:05 2018

@author: qijingzhao
"""
import numpy as np
from .cos_models import *
from scipy.integrate import quad
import scipy.constants as sc
import scipy.stats as stats
import pandas as pd
from .tools import simp_err,savetxt
from astroML.density_estimation import FunctionDistribution
#====================constants=================================
Mpc=sc.parsec*1e6
Msun=1.98847*1e30
#--------------------------------------------------------------
f0=200.0;S0=1.449*1e-52
p1,p2=-4.05,-0.69;a1,a2=185.62,232.56
b1,b2,b3,b4,b5,b6=31.18,-64.72,52.24,-42.16,10.17,11.53
c1,c2,c3,c4=13.58,-36.46,18.56,27.43
f_lower=1.0
#-------------------------------------------------------------

#==============================================================
class ET(object):
    def __init__(self,model_name='LCDM', params=[0.308,0.678],GW_type=0.03):
        self.ll=globals().get('%s'%model_name)(*params)
        self._GW_type=GW_type
        self._gw_choice()
    
    def _gw_choice(self):
        if self._GW_type=='BHNS':
            print('The GW events you choose is %s'%self._GW_type)
        elif self._GW_type=='NSNS':
            print('The GW events you choose is %s'%self._GW_type)
        elif self._GW_type=='BHBH':
            print('The GW events you choose is %s'%self._GW_type)
        elif 0<self._GW_type<1:
            print('The ratio between BHNS and BNS events = %s'%self._GW_type)
        else:
            raise NameError('The type of GW your choice is wrong!\n\n\
        Please choose from ["BHNS","NSNS","BHBH"]')

#=====================================================

#=====================================================
    @staticmethod
    def F1plus(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
        return np.sqrt(3)/2.0*(a-b)
    
    @staticmethod
    def F1mul(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
        return np.sqrt(3)/2.0*(a+b)
    
    @staticmethod
    def F2plus(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*(phi+2*np.pi/3))*np.cos(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*(phi+2*np.pi/3))*np.sin(2*psi)
        return np.sqrt(3)/2.0*(a-b)
    
    @staticmethod
    def F2mul(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*(phi+2*np.pi/3))*np.sin(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*(phi+2*np.pi/3))*np.cos(2*psi)
        return np.sqrt(3)/2.0*(a+b)
    
    @staticmethod
    def F3plus(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*(phi+4*np.pi/3))*np.cos(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*(phi+4*np.pi/3))*np.sin(2*psi)
        return np.sqrt(3)/2.0*(a-b)
    
    @staticmethod
    def F3mul(theta,phi,psi):
        a=(1+np.cos(theta)**2)*np.cos(2*(phi+4*np.pi/3))*np.sin(2*psi)/2.0
        b=np.cos(theta)*np.sin(2*(phi+4*np.pi/3))*np.cos(2*psi)
        return np.sqrt(3)/2.0*(a+b)
    @staticmethod
    def Sh(f):
        x=f/f0
        fz=1+b1*x+b2*x**2+b3*x**3+b4*x**4+b5*x**5+b6*x**6
        fm=1+c1*x+c2*x**2+c3*x**3+c4*x**4
        return S0*(x**p1+a1*x**p2+a2*fz/fm)
    
    def Int_sh(self,f):
        return f**(-7/3)/self.Sh(f)

# =============================================================================
#=============================================================================
    def __snr(self,z,event='BHNS'):
        if event=='BHNS':
            m1_range=[1,2]
            m2_range=[3,10]
        elif event=='NSNS':
            m1_range=[1,2]
            m2_range=[1,2]
        elif event=='BHBH':
            m1_range=[3,10]
            m2_range=[3,10]
        rho=0.0
        while rho<=8.0:
            m1=np.random.uniform(m1_range[0],m1_range[1])*Msun
            m2=np.random.uniform(m2_range[0],m2_range[1])*Msun
            M=m1+m2
            eta=m1*m2/M**2
            Mc_phys=M*eta**(3.0/5.0)
            Mc_obs=(1+z)*Mc_phys
            theta=np.random.uniform(0,np.pi)
            phi=np.random.uniform(0,2*np.pi)
            iota=0.0
            psi=np.pi/4
            M_obs=(1+z)*(m1+m2)
            f_upper=2/np.power(6,1.5)/2/np.pi/M_obs/sc.G*sc.c**3
            A1=np.sqrt(self.F1plus(theta,phi,psi)**2*(1+np.cos(iota)**2)**2+4*self.F1mul(theta,phi,psi)**2*
                       np.cos(iota)**2)*np.sqrt(5*np.pi/96)*np.pi**(-7/6)*Mc_obs**(5/6)/self.ll.lum_dis_z(z)/Mpc*\
                       np.power(sc.G,5/6)*np.power(sc.c,-1.5)
            A2=np.sqrt(self.F2plus(theta,phi,psi)**2*(1+np.cos(iota)**2)**2+4*self.F2mul(theta,phi,psi)**2*
                       np.cos(iota)**2)*np.sqrt(5*np.pi/96)*np.pi**(-7/6)*Mc_obs**(5/6)/self.ll.lum_dis_z(z)/Mpc*\
                       np.power(sc.G,5/6)*np.power(sc.c,-1.5)
            A3=np.sqrt(self.F3plus(theta,phi,psi)**2*(1+np.cos(iota)**2)**2+4*self.F3mul(theta,phi,psi)**2*
                       np.cos(iota)**2)*np.sqrt(5*np.pi/96)*np.pi**(-7/6)*Mc_obs**(5/6)/self.ll.lum_dis_z(z)/Mpc*\
                       np.power(sc.G,5/6)*np.power(sc.c,-1.5)
            ET1=4*quad(self.Int_sh,f_lower,f_upper)[0]*A1**2
#            ET1=2*np.sqrt(sc.G**(5/3)*sc.c**(-3)*A1**2*Nconst)
            ET2=4*quad(self.Int_sh,f_lower,f_upper)[0]*A2**2
            ET3=4*quad(self.Int_sh,f_lower,f_upper)[0]*A3**2
            rho=np.sqrt(ET1+ET2+ET3)
#        self.count+=1
#        print(self.count)
        return z,m1/Msun,m2/Msun,theta,phi,rho
#====================分布函数=================================
    def Pz(self,z):
        def Rz(z):
            if z<=1:
                r=1.+2.*z
            elif 1<z<5:
                    r=3.*(5.-z)/4.
            else:
                r=0.0
            return r
        R=np.vectorize(Rz)
        return 4.*np.pi*self.ll.d_z(z)**2*R(z)/self.ll.hubz(z)/(1.0+z)

    def ET_z(self,zz,rand='normal'):
        Dd=np.zeros(len(zz))-1
        while (Dd<=0).any():
            if type(self._GW_type)==float:
                N=len(zz)
                n_BHNS=round(self._GW_type*N)
                index_BHNS=np.random.choice(np.where(zz<5)[0],n_BHNS, replace=False)
                BNS_zz=np.delete(zz,index_BHNS)
                print('The number of BHNS events is %s.'%n_BHNS)
                print('The number of BNS events is %s.'%len(BNS_zz))
                zh,m1h,m2h,thetah,phih,rhoh=np.vectorize(self.__snr)(zz[index_BHNS],event='BHNS')
                zn,m1n,m2n,thetan,phin,rhon=np.vectorize(self.__snr)(BNS_zz,event='NSNS')
                self.z=np.append(zh,zn)
                self.m1=np.append(m1h,m1n)
                self.m2=np.append(m2h,m2n)
                self.theta=np.append(thetah,thetan)
                self.phi=np.append(phih,phin)
                self.rho=np.append(rhoh,rhon)
            else:
                self.z,self.m1,self.m2,self.theta,self.phi,self.rho=np.vectorize(self.__snr)(zz,self._GW_type)
            DL=self.ll.lum_dis_z(self.z)
            DL_err=np.sqrt((2.*DL/self.rho)**2+(0.05*self.z*DL)**2)
            if rand=='normal':
                DL_mean=np.random.normal(DL,DL_err)
            elif rand=='1sigma':
                DL_mean=stats.truncnorm(-1,1,loc=DL,scale=DL_err).rvs()
            elif rand=='None':
                DL_mean=DL
            else:
                print('The input of random is wrong.')
            Dd=DL_mean-DL_err
        self.DL=DL_mean
        self.DL_err = DL_err
        return self.z ,DL_mean,DL_err


    def ET_default(self,zlow=0,zup=5,num=1000,rand='normal'):
#        self.count=0
#        if type(self._GW_type)==str:
#            if zup>self.z_max: zup=self.z_max
        zzn=FunctionDistribution(self.Pz,zlow,zup,num*2).rvs(num)
        zz,DL_mean,DL_err = self.ET_z(zzn,rand)
        return zz,DL_mean,DL_err

    def save_fulldata(self,path_file_name):
        st=['#z','m1','m2','theta','phi','snr']
        try:
            if self.z.any():
                data=(self.z,self.m1,self.m2,self.theta,self.phi,self.rho)
                dc=dict(zip(st,data))
                df=pd.DataFrame(dc)
                if path_file_name[-3:]=='lsx':
                    df.to_excel(path_file_name,index=False)
                elif path_file_name[-3:]=='txt':
                    df.to_csv(path_file_name,index=False,sep=' ')
                else:
                    df.to_excel(path_file_name+'.xlsx',index=False)
        except AttributeError:
            print('Please run the GW_z function firstly!')
    
    def save_DL(self,path_file_name):
#        st=['#z','DL','DL_err']
        try:
            if self.z.any():
                savetxt(path_file_name,[self.z,self.DL,self.DL_err])
#                data=(self.z,self.DL,self.DL_err)
##                dc=dict(zip(st,data))
#                dc=dict(data)
#                df=pd.DataFrame(dc)
#                if path_file_name[-3:]=='lsx':
#                    df.to_excel(path_file_name,index=False)
#                elif path_file_name[-3:]=='txt':
#                    df.to_csv(path_file_name,index=False,sep=' ')
#                else:
#                    df.to_csv(path_file_name,index=False,sep=' ')
        except AttributeError:
            print('Please run the GW_z function firstly!')
    
class Ligo(object):
    def __init__(self,Gam,Lam,ll,kesi=np.pi/2,psi=np.pi/4,iota=0):
        self.Gam = Gam
        self.Lam = Lam
        self.kesi = kesi
        self.psi = psi
        self.iota = iota
        self.ll=ll
        
    def at(self,alpha,delta,ang):
        Gam,Lam = self.Gam,self.Lam
        a=np.sin(2*Gam)*(3-np.cos(2*Lam))*(3-np.cos(2*delta))*np.cos(2*(alpha-ang))/16
        b=np.cos(2*Gam)*np.sin(Lam)*(3-np.cos(2*delta))*np.sin(2*(alpha-ang))/4
        c=np.sin(2*Gam)*np.sin(2*Lam)*np.sin(2*delta)*np.cos(alpha-ang)/4
        d=np.cos(2*Gam)*np.cos(Lam)*np.sin(2*delta)*np.sin(alpha-ang)/2
        e=np.sin(2*Gam)*np.cos(Lam)**2*np.cos(delta)**2*3/4
        return a-b+c-d+e
    

    def bt(self,alpha,delta,ang):
        Gam,Lam = self.Gam,self.Lam
        a=np.cos(2*Gam)*np.sin(Lam)*np.sin(delta)*np.cos(2*(alpha-ang))
        b=np.sin(2*Gam)*(3-np.cos(2*Lam))*np.sin(delta)*np.sin(2*(alpha-ang))/4
        c=np.cos(2*Gam)*np.cos(Lam)*np.cos(delta)*np.cos(alpha-ang)
        d=np.sin(2*Gam)*np.sin(2*Lam)*np.cos(delta)*np.sin(alpha-ang)/2
        return a+b+c+d
    

    def Fplus(self,alpha,delta,ang):
        kesi = self.kesi
        psi = self.psi
        return np.sin(kesi)*(self.at(alpha,delta,ang)*np.cos(2*psi)+
                      self.bt(alpha,delta,ang)*np.sin(2*psi))
    
    def Fmul(self,alpha,delta,ang):
        kesi = self.kesi
        psi = self.psi
        return np.sin(kesi)*(self.bt(alpha,delta,ang)*np.cos(2*psi)-
                      self.at(alpha,delta,ang)*np.sin(2*psi))
    
    def AA(self,alpha,delta,ang,Mc_obs,z):
        return np.sqrt(self.Fplus(alpha,delta,ang)**2*(1+np.cos(self.iota)**2)**2+4*self.Fmul(alpha,delta,ang)**2*
                       np.cos(self.iota)**2)*np.sqrt(5*np.pi/96)*np.pi**(-7/6)*Mc_obs**(5/6)/self.ll.lum_dis_z(z)/Mpc
    
    
    @staticmethod
    def Sh(f):
        f0,S0=215,1e-49
        x=f/f0
        return S0*(x**(-4.14)-5*x**(-2)+111*(1-x**2+0.5*x**4)/(1+0.5*x**2))
    

    def Ncont(self):
        Nint=lambda f:f**(-7/3)/self.Sh(f)
        return quad(Nint,20,2000)[0]
    
    def snr(self,alpha,delta,ang,Mc_obs,z):
        return 2*np.sqrt(sc.G**(5/3)*sc.c**(-3)*self.AA(alpha,delta,ang,Mc_obs,z)**2*self.Ncont())

class Virgo(Ligo):
    def __init__(self,Gam,Lam,ll,kesi=np.pi/2,psi=np.pi/4,iota=0):
        self.Gam = Gam
        self.Lam = Lam
        self.psi = psi
        self.kesi = kesi
        self.iota = iota
        self.ll=ll
    
    @staticmethod
    def Sh(f):
        f0,S0=720,1e-47
        x=f/f0
        a=np.log(x)**2*(3.2+1.08*np.log(x)+0.13*np.log(x)**2)
        b=0.73*np.log(x)**2
        c=2.67e-7*x**(-5.6)+0.59*x**(-4.1)*np.exp(-a)+0.68*x**(5.34)*np.exp(-b)
        return S0*c

class LgVg(object):
    def __init__(self,model_name='LCDM', params=[0.308,0.678],m1_range=[3,100],m2_range=[3,100]):
        self.ll=globals().get('%s'%model_name)(*params)
#        self._GW_type=GW_type
#        self._gw_choice()
        self.model_name=model_name
        self.params=params
        self._m1_range=m1_range
        self._m2_range=m2_range
    
#    def _gw_choice(self):
#        if self._GW_type=='BHNS':
#            self._m1_range=[1,2]
#            self._m2_range=[3,10]
#        elif self._GW_type=='NSNS':
#            self._m1_range=[1,2]
#            self._m2_range=[1,2]
#        elif self._GW_type=='BHBH':
#            self._m1_range=[3,10]
#            self._m2_range=[3,10]
#        else:
#            raise NameError('The type of GW your choice is wrong!\n\n\
#        Please choose from ["BHNS","NSNS","BHBH"]')
# =============================================================================
#=============================================================================
    def __snr(self,z):
        H1=Ligo(171.8*np.pi/180,46.45*np.pi/180,self.ll)
        H1longitude = (119+24/60+27.6/60**2)*np.pi/180
        
        L1=Ligo(243*np.pi/180,30.56*np.pi/180,self.ll)
        L1longitude = (90+46/60+27.3/60**2)*np.pi/180
        
        V1=Virgo(116.5*np.pi/180,43.63*np.pi/180,self.ll)
        V1longitude = (10+30/60+16/60**2)*np.pi/180
        
        snrH1,snrL1,snrV1=0.0,0.0,0.0
        mr=0
        while snrH1<=8.0 or snrL1<=8.0 or snrV1<=8.0 or mr<0.5 or mr>2:
            m1=np.random.uniform(self._m1_range[0],self._m1_range[1])*Msun
            m2=np.random.uniform(self._m2_range[0],self._m2_range[1])*Msun
            M=m1+m2
            eta=m1*m2/M**2
            Mc_phys=M*eta**(3.0/5.0)
            Mc_obs=(1+z)*Mc_phys
            mr=m1/m2
            
            alpha=np.random.uniform(0,np.pi)
            delta=np.random.uniform(0,2*np.pi)
            
            ang=np.random.uniform(0,2*np.pi)
            snrH1=H1.snr(alpha,delta,ang,Mc_obs,z)
            
            L1_ang=ang+(L1longitude-H1longitude)
            snrL1=L1.snr(alpha,delta,L1_ang,Mc_obs,z)
            
            V1_ang=ang-(V1longitude+H1longitude)
            snrV1=V1.snr(alpha,delta,V1_ang,Mc_obs,z)
        
        rho=np.sqrt(snrH1**2+snrL1**2+snrV1**2)

        self.count+=1
#        print(self.count)
        return z,m1/Msun,m2/Msun,alpha,delta,snrH1,snrL1,snrV1,rho
#====================分布函数=================================
    def Pz(self,z):
        def Rz(z):
            if z<=1:
                r=1.+2.*z
            elif 1<z<5:
                    r=3.*(5.-z)/4.
            else:
                r=0.0
            return r
        R=np.vectorize(Rz)
        return 4.*np.pi*self.ll.d_z(z)**2*R(z)/self.ll.hubz(z)/(1.0+z)

    def GW_z(self,zz,rand='normal'):
        self.count=0
        self.z,self.m1,self.m2,self.alpha,self.delta,self.snrH1,self.snrL1,self.snrV1,self.rho\
        =np.vectorize(self.__snr)(zz)
        DL=np.vectorize(self.ll.lum_dis_z)(zz)
        DL_err=np.sqrt((2.*DL/self.rho)**2+(0.05*zz*DL)**2)
        if rand=='normal':
            DL_mean=np.random.normal(DL,DL_err)
        elif rand=='1sigma':
            DL_mean=stats.truncnorm(-1,1,loc=DL,scale=DL_err).rvs()
        elif rand=='None':
            DL_mean=DL
        else:
            print('The input of random is wrong.')
        self.DL=DL_mean
        self.DL_err=DL_err
        return DL_mean,DL_err
    
    def save_fulldata(self,path_file_name):
        st=['#z','m1','m2','alpha','delta','snrH1','snrL1','snrV','snr']
        try:
            if self.z.any():
                data=(self.z,self.m1,self.m2,self.alpha,self.delta,self.snrH1,self.snrL1,self.snrV1,self.rho)
                dc=dict(zip(st,data))
                df=pd.DataFrame(dc)
                if path_file_name[-3:]=='lsx':
                    df.to_excel(path_file_name,index=False)
                elif path_file_name[-3:]=='txt':
                    df.to_csv(path_file_name,index=False,sep=' ')
                else:
                    df.to_excel(path_file_name+'.xlsx',index=False)
        except AttributeError:
            print('Please run the GW_z function firstly!')
    
    def save_DL(self,path_file_name):
        st=['#z','DL','DL_err']
        try:
            if self.z.any():
                data=(self.z,self.DL,self.DL_err)
                dc=dict(zip(st,data))
                df=pd.DataFrame(dc)
                if path_file_name[-3:]=='lsx':
                    df.to_excel(path_file_name,index=False)
                elif path_file_name[-3:]=='txt':
                    df.to_csv(path_file_name,index=False,sep=' ')
                else:
                    df.to_csv(path_file_name,index=False,sep=' ')
        except AttributeError:
            print('Please run the GW_z function firstly!')
    
    def Ligo_default(self,zlow=0.0,zup=1.0,num=800,rand='normal'):
        zzn=zzn=FunctionDistribution(self.Pz,zlow,zup,num).rvs(num)
        DL,DL_s=self.GW_z(zzn,rand)
        return zzn,DL,DL_s


#class LISA(object):
#    
#    def __init__(self,Omegam=0.3,h=0.7,random=True):
#        self.ll=LCDM(Omegam,h)
#        self.H0 = h*1e2
#        self.Hz=self.ll.hubz
#        self.dc=self.ll.d_z
#        self.dlz=np.vectorize(self.ll.lum_dis_z)
#        self.random = random
##        self.DL_err = DL_err
#    
#    def redshift_distribution(self,number):
#        Ngw=np.asarray([3.6,10.3,9.3,7.5,4.7,2.8,1.2,0.4,0.2,0.0])
#        ratio=Ngw/np.sum(Ngw)
#        true_n=number-1
#        addn=0
#        bins=np.arange(0,11,1)
#        while true_n<number:
#            fb=map(int,map(round,(number+addn)*ratio*100))
#            true_n=sum(fb)
#            addn=addn+1
#        zzn=np.random.uniform(0.31,bins[1],fb[0])
#        for i in range(1,len(bins)-1):
#            zn=np.random.uniform(bins[i],bins[i+1],fb[i])
##            print(zn,i)
#            zzn=np.append(zzn,zn)
##        if self.endpoint:
##            zzn=np.append(zzn,[0.5,6])
#        redshift=np.random.choice(zzn,number,replace=False)
#        return np.sort(redshift)
#    
#    def DL(self,z):
#        return self.dlz(z)
#    
#    def DL_s(self,z):
#        lens=self.DL(z)*0.066*np.power((1.0-np.power(1.+z,-0.25))/0.25,1.8)
##        vz=self.dlz(z)*Mpc*(1.+(1.+z)/self.dlz(z)*self.ll.D_H())*(500.0*1e3/c0)
#        return lens
#
#    def dp_lens2(self,z,mud,mud_err=0.05):
#        Dl_th,DL_s=self.DL(z),self.DL_s(z)
#        A_obs=np.sqrt(mud)/Dl_th
#        A_s=A_obs*DL_s/Dl_th
#        def dps(zs,A,mu):
#            return np.sqrt(mu)/A/(1+zs)
#        dp,dp_s=simp_err(dps,[z,A_obs,mud],[z*0.0,A_s,mud*mud_err])
#        return dp/self.ll.D_H(), dp_s/self.ll.D_H()
#    
#    def gw(self,number):
#        z=self.redshift_distribution(number)
#        dl_err = self.DL_s(z)
#        dl = self.DL(z)
#        if self.random:
#            dll=stats.truncnorm(-1.0,1.0,loc=dl, scale=dl_err).rvs()
#        else:
#            dll=dl
#        return z,dll,dl_err
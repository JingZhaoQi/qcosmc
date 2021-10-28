# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:07:21 2017

@author: jingzhao
"""
__package__ = 'qcosmc'
import numpy as np
from scipy.integrate import quad,odeint
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,splev,splrep
#import transfer_func as tf
from scipy.optimize import fsolve
from scipy.constants import c,m_p,G,parsec
import scipy.constants as sc
from .Decorator import vectorize
#from .SN import SN_likelihood
#import os
Mpc=parsec*1e6
#dataDir=os.path.dirname(os.path.abspath(__file__))+'/data/'
#like = SN_likelihood(dataDir+'full_long.dataset')
class LCDM(object):
    def __init__(self,Om0,h=0.674,OmK=0.0,Omr0=None,Ob0h2=0.02225,ns=0.96,sigma_8 = 0.8):
        self.Om0 = np.float64(Om0)
        self.h = np.float64(h)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.sigma_8 = np.float64(sigma_8)
        if Omr0:
            self.Omega_r=0.0
        else:
            self.Omega_r = self.Omega_r0
        
    @property
    def Omega_r0(self):
        T_CMB=2.7255
        z_eq=2.5e4*self.Om0*self.h**2*(T_CMB/2.7)**(-4)
        return self.Om0/(1.0+z_eq)

    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3+self.Omega_r*(1+z)**4+self.OmK*(1+z)**2+
                       (1.-self.Om0-self.Omega_r-self.OmK))

    def wz(self,z):
        return z-z-1.
    
    def Hz(self,z):
        return self.hubz(z)*self.h*1e2       #Mpc
    
    def Drift(self,z,yr):
        """
        Drift as a function of redshift z and observed years

        Parameters
        ----------
        z : redshift
        yr : observed time in units of year

        Returns drift in units of cm/s
        
        c~m/s, H0~km/s/Mpc, year~s
        c*H0*year*1e5 ----> cm/s
        """
        year=yr*sc.year # s
        H0=self.h*1e2/Mpc #km/s/Mpc
        return c*H0*year*(1-self.hubz(z)/(1+z))*1e5 # cm/s
	
    @vectorize
    def Drift_ELT(self,z):
        if 2.<=z and z<=4.:
            x=-1.7
        elif z>4.0:
            x=-0.9
        else:
            x=0
            
        SN=3000.
        N=30.
        dd=1.35*(2370/SN)/np.sqrt(N/30)*((1+z)/5)**x
        return dd
    
    def E2(self,z):
        return self.hubz(z)**2

    def invhub(self,z):
        return 1./self.hubz(z)

    def huba(self,a):
        return self.hubz(1./a-1)

    def hubpa(self,a):
        """
        Derivative of dimensionless hubble constant w.r.t. a
        """
        return derivative(self.huba,a,dx=1e-6)

    @property
    def D_H(self):
        """
        Hubble distance in units of MPc (David Hogg arxiv: astro-ph/9905116v4)
        """
        return c/1e5/self.h

    @vectorize
    def co_dis_z(self,z1):
        """
        Line of sight comoving distance as a function of redshift z
        as described in David Hogg paper
        in units of MPc
        """
        if self.OmK==0.0:
            dis=self.D_H*quad(self.invhub,0,z1)[0]
        elif self.OmK<0.0:
            dis=self.D_H*np.sin(np.sqrt(-self.OmK)*quad(self.invhub,0,z1)[0])/np.sqrt(-self.OmK)
        else:
            dis=self.D_H*np.sinh(np.sqrt(self.OmK)*quad(self.invhub,0,z1)[0])/np.sqrt(self.OmK)
        return dis
    
    def d_z(self,z1):
        """
        Line of sight comoving distance as a function of redshift z
        as described in David Hogg paper
        dimensionless
        """
        return self.co_dis_z(z1)/self.D_H

	#angular diameter distance in units of MPc
    def ang_dis_z(self,z):
        """
        Angular diameter distance as function of redshift z
        in units of MPc
        """
        d_c = self.co_dis_z(z)/(1.+z)
        return d_c


#  ======CMB R, l_A, z_start
    @property
    def zs_z(self):
        """
        Decoupling redshift as a constant
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        """
        obh2=self.Ob0h2
        omh2=self.Om0*self.h**2
        g1=0.0783*obh2**(-0.238)/(1+39.5*obh2**(0.763))
        g2=0.56/(1+21.1*obh2**(1.81))
        z_start=1048.0*(1+0.00124*obh2**(-0.738))*(1.+g1*omh2**g2)
        return z_start
    
    def rs_a(self,a):
        """
        Comoving sound horizon r_s as a function of a
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        Rb=3/(4*Omega_r*h**2)
        """
        T_CMB=2.7255
        obh2=self.Ob0h2
        Rb=31500.0*(T_CMB/2.7)**(-4.0)
#        orh2=2.469e-5
        def soundH(a):
            return 1.0/(a**2*self.huba(a)*np.sqrt(3.0*(1.0+obh2*a*Rb)))
#            return 1.0/(a**2*self.huba(a)*np.sqrt(3.0*(1.0+(3.0*obh2/4.0/orh2)*a)))
        rs=self.D_H*quad(soundH,1e-8,a)[0]
        return rs
    
    def rs_z(self,z):
        """
        Comoving sound horizon r_s as a function of z
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        Rb=3/(4*Omega_r*h**2)
        """
#        T_CMB=2.7255
#        obh2=self.Ob0h2
#        Rb=31500.0*obh2*(T_CMB/2.7)**(-4)
#        def sH(z):
#            cs=1/np.sqrt(3.0*(1.0+Rb/(1.0+z)))
#            return cs/self.hubz(z)
#        rsz=self.D_H*quad(sH,z,np.inf)[0]
#        return rsz
        return self.rs_a(1/(1+z))

    
    @property
    def l_A(self):
        """
        The acoustic scale l_A as a constant
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        """
        zs=self.zs_z
        return np.pi*(1+zs)*self.ang_dis_z(zs)/self.rs_z(zs)
#        return np.pi*self.co_dis_z(zs)/self.rs_a(1.0/(1+zs))
       
    @property
    def Rth(self):
        """
        the shift parameter R as a constant
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        """
        zs=self.zs_z
        return np.sqrt(self.Om0)*(1+zs)*self.ang_dis_z(zs)/self.D_H
    
    @property
    def zd(self):
        """
        the baryon drag epoch z
        JCAP 1512 (2015) 12, 022
        or ArXiv:1609.08220
        """
        omh2=self.Om0*self.h**2
        obh2=self.Ob0h2
        b1=0.313*omh2**(-0.419)*(1+0.607*omh2**(0.674))
        b2=0.238*omh2**(0.223)
        zzd=1291*omh2**(0.251)*(1+b1*obh2**b2)/(1+0.659*omh2**(0.828))
        return zzd
    
    @property
    def rd(self):
        return self.rs_z(self.zd)
    
    def Dv(self,z):
        return ((1+z)**2*self.ang_dis_z(z)**2*z*c/1e3/self.Hz(z))**(1/3)
    
    def rs_over_Dv(self,z):
        return self.rd/self.Dv(z)
    
    def DM_rd(self,z,rd_f=147.78):
        return (1+z)*self.ang_dis_z(z)*rd_f/self.rd
    
    def H_rd(self,z,rd_f=147.78):
        return self.Hz(z)*self.rd/rd_f
    
    def dA_over_DV(self,z):
        dA=self.co_dis_z(self.zs_z)
        DV=np.power(self.co_dis_z(z)**2*z*self.D_H/self.hubz(z),1/3)
        return dA/DV
    

    @vectorize
    def co_dis_z2(self,zl,zs):
        """
        Angular diameter distance as function of two redshift zl, zs using for Lensing
        in units of MPc
        """
        if self.OmK==0.0:
            dis=self.D_H*quad(self.invhub,zl,zs)[0]
        elif self.OmK<0.0:
            dis=self.D_H*np.sin(np.sqrt(-self.OmK)*quad(self.invhub,zl,zs)[0])/np.sqrt(-self.OmK)
        else:
            dis=self.D_H*np.sinh(np.sqrt(self.OmK)*quad(self.invhub,zl,zs)[0])/np.sqrt(self.OmK)
        return dis
    
    def ang_dis_z2(self,zl,zs):
        return self.co_dis_z2(zl,zs)/(1.0+zs)
    
    def lensing(self,zl,zs):
        """
        The distance ratio of D_ls/D_s
        The Astrophysical Journal, 806:185 (12pp), 2015 June 20
        doi:10.1088/0004-637X/806/2/185
        COSMOLOGY WITH STRONG-LENSING SYSTEMS
        """
        return self.co_dis_z2(zl,zs)/self.co_dis_z2(0,zs)

	
    def lum_dis_z(self,z):
        """
        luminosity distance in units of MPc
        """
        d_l = self.co_dis_z(z)*(1.+z)
        return d_l
    
    def MC_DL(self,z,step=0.1):
        zz=np.arange(0,z.max()+step,step)
        return splev(z,splrep(zz,self.lum_dis_z(zz)))
    
    def MC_DA(self,z,step=0.1):
        zz=np.arange(0,z.max()+step,step)
        return splev(z,splrep(zz,self.ang_dis_z(zz)))
    
    def mu(self,z):
        dl=self.lum_dis_z
        return 5. * np.log10(dl(z)) + 25.

#==============FRB The Astrophysical Journal Letters, 860:L7 (6pp), 2018 June 10===================
    @vectorize
    def Xz(self,z):
        y1=1
        y2=4-3*y1
        XeH=1
        XeHe=1
        Xz=(3/4)*y1*XeH+(1/8)*y2*XeHe
        def xzint(z):
            return Xz*(1+z)/self.hubz(z)
        return quad(xzint,0,z)[0]
    
    def Xz_MC(self,z):
        return splev(z,splrep(self.zz,self.Xz(self.zz)))
    
    def DM(self,z):
        f_IGM=0.83
        return 3*1e2*self.Ob0h2/self.h*f_IGM*c*(1e3/Mpc/1e6/parsec)/(8*np.pi*m_p*G)*self.Xz_MC(z)
#=========================================================================================================

#    def addSN(self):
#        zs = like.get_redshifts()
#        self.chi2=self.chi2+like.loglike(self.MC_DA(zs))*2
        
    @vectorize
    def time_delay_dis(self,zl,zs):
        D_ds = self.ang_dis_z2(zl,zs)
        return self.ang_dis_z(zl)*self.ang_dis_z(zs)/D_ds*(1+zl)


    @property
    def t_H(self):
        return 9.78461942321705/self.h


    def om_a(self,a):
        """
        density parameter \Omega_{m} for matter as a function of scale factor a
        """
        return self.Om0*a**(-3)/self.huba(a)**2.

    def om_z(self,z):
        """
        density parameter \Omega_{m} for matter as a function of z
        """
        return self.om_a(1./(1.+z))
		
    def Om_diag(self,z):
        """
        Om diagnostic
        See Varun Sahni et. al., Phys. Rev. D. 78, 103502 (2008) for more details
        """
        x = 1+z
        return (self.hubz(z)**2. - 1)/(x**3. - 1.)

    def L1(self,z):
        """
        The derivation of Om_diag
        """
        return derivative(self.Om_diag,z,dx=1e-9)/((1.+z)**6)

    def qz(self,z):
        dEz=derivative(self.hubz,z,dx=1e-3)
        q=(1.+z)*dEz/self.hubz(z)-1.
        return q

    @vectorize
    def lookback_time_z(self,z):
        """
        Lookback time as a function of redshift z
        in units of billion years
        """
        def integrand(z):
            return self.invhub(z)/(1.+z)
        lt = quad(integrand,0,z)[0]
        return self.t_H*lt

    @property
    def age_by(self):
        """
        Age of the Universe in units of billion years
        """
        def integrand(z1):
            return self.invhub(z1)/(1.+z1)
        t0 = quad(integrand,0,np.inf)[0]
        return self.t_H*t0
		
#==========================statefinder======================================================
    """
    statefinder in Ref PHYSICAL REVIEW D 83, 043501 (2011)
    """
#============================A(n)========================================================
	#--------------------A(n)_z-----------------------------
    def A2(self,z):
        return -self.qz(z)

    
    def A3(self,z):
        def A3q(z):
            return self.E2(z)*(1.+self.qz(z))
        dE2=derivative(A3q,z,dx=1e-3)
        return (1.+z)*dE2/self.E2(z)-3.*self.qz(z)-2.
        
    def A4(self,z):
        def A4q(z):
            return self.hubz(z)**3*(2.+3.*self.qz(z)+self.A3(z)) 
        dE3=derivative(A4q,z,dx=1e-3)
        return -(1.+z)*dE3/self.hubz(z)**3+4.*self.A3(z)+3.*self.qz(z)*(self.qz(z)+4.)+6.

    def A5(self,z):
        def A5q(z):
            return self.hubz(z)**4*(self.A4(z)-4.*self.A3(z)-3.*self.qz(z)*(self.qz(z)+4)-6.)
        dE4=derivative(A5q,z,dx=1e-3)
        return -(1.+z)*dE4/self.hubz(z)**4+5.*self.A4(z)-10.*self.A3(z)*(self.qz(z)+2.)-30.*self.qz(z)*(self.qz(z)+2.)-24.
#==========================S(n)==========================================================
        #------------------S(n)_z--------------------------------
    def S2(self,z):
        return self.A2(z)+3./2.*self.om_z(z)
    def S3(self,z):
        return self.A3(z)
    def S4(self,z):
        return self.A4(z)+3.*(1.+self.qz(z))
    def S5(self,z):
        return self.A5(z)-2.*(4.+3.*self.qz(z))*(1.+self.qz(z))

    #-----------------s(n)^(2)-------------------------------
    def S3_2(self,z):
        return (self.S3(z)-1.)/(3.*self.qz(z)-3./2.)
    def S4_2(self,z):
        return (self.S4(z)-1.)/(9.*self.qz(z)-9./2.)
    
    def Statefinder_r(self,z):
        return self.A3(z)
    def Statefinder_s(self,z):
        return (self.Statefinder_r(z)-1)/(self.qz(z)-0.5)/3
    
    def jerk_j(self,z):
        return self.A3(z)
    def snap_s(self,z):
        return self.A4(z)
    def lerk_l(self,z):
        return self.A5(z)
       
#====================================================================================================
#============================growth parameter ee(z)========================
    def gr_lcdm(self,z):
        om=0.3*(1.+z)**3/(0.3*(1.+z)**3.+(1.-0.3))
        w=-1.
        r=3./(5.-w/(1.-w))+3.*(1.-w)*(1.-3./2.*w)/(125.*(1.-6./5.*w)**3)*(1.-om)
        return om**r

    def ee(self,z):
        om=self.om_z(z)
        w=self.wz(z)
        r=3./(5.-w/(1.-w))+3.*(1.-w)*(1.-3./2.*w)/(125.*(1.-6./5.*w)**3)*(1.-om)
        return om**r/self.gr_lcdm(z)
    """
    end statefinder
    """

    a11 = np.linspace(0.001,1,1000)

    def deriv(self,y,a):
        return [ y[1], -(3./a + self.hubpa(a)/self.huba(a))*y[1] + 1.5*self.om_a(a)*y[0]/(a**2.)]

    def sol1(self):
        yinit = (0.001,1.)
        return odeint(self.deriv,yinit,self.a11)

    def D_p(self,a):
        yn11=self.sol1()[:,0]
        ynn11=UnivariateSpline(self.a11,yn11,k=3,s=0)
        return ynn11(a)

    def D_plus_a(self,a):
        return self.D_p(a)/self.D_p(1.0)

    def D_plus_z(self,z):
        """
        Normalized solution for the growing mode as a function of redshift
        """
        return self.D_plus_a(1./(1.+z))

    def growth_rate_a(self,a):
        d = self.sol1()[:,:]
        d1 = d[:,0]
        dp = d[:,1]
        gr1 = UnivariateSpline(self.a11,self.a11*dp/d1,k=3,s=0)
        return gr1(a)

    def growth_rate_z(self,z):
        """
        Growth Rate f = D log(Dplus)/ D Log(a) as a function of redshift
        """
        return self.growth_rate_a(1./(1.+z))

    def fsigma8z(self,z):
        """
        fsigma_{8} as a function of redshift
        """
        return self.growth_rate_z(z)*self.D_plus_z(z)*self.sigma_8
	
	# Defining window function

#    def Wf(self,k):
#        return 3.*(np.sin(k*8.) - (k*8.)*np.cos(k*8.))/(k*8.)**3.
#
#    def integrand_bbks(self,k):
#        return k**(self.ns + 2.)*tf.Tbbks(k,self.Om0,self.Ob0,self.h)**2.*(self.Wf(k))**2.
#
#    def integrand_wh(self,k):
#        return k**(self.ns + 2.)*tf.Twh(k,self.Om0,self.Ob0,self.h)**2.*(self.Wf(k))**2.
#
#    def A0bbks(self):
#        return (2.0*np.pi**2*self.sigma_8**2.)/(quad(self.integrand_bbks,0,np.inf)[0])
#
#    def A0wh(self):
#        return (2.0*np.pi**2*self.sigma_8**2.)/(quad(self.integrand_wh,0,np.inf)[0])
#
#    def Pk_bbks(self,k,z):
#        """
#        Matter Power Spectra Pk in units if h^{-3}Mpc^{3} as a function of k in units of [h Mpc^{-1}]
#        and z;
#        Transfer function is taken to be BBKS
#        Ref: Bardeen et. al., Astrophys. J., 304, 15 (1986)
#        """
#        return self.A0bbks()*k**self.ns*tf.Tbbks(k,self.Om0,self.Ob0,self.h)**2.*self.D_plus_z(z)**2.
#
#    def Pk_wh(self,k,z):
#        """
#		Matter Power Spectra Pk in units if h^{-3}Mpc^{3} as a function of k in units of [h Mpc^{-1}]
#		and z;
#		Transfer function is taken to be Eisenstein & Hu 
#		Ref: Eisenstein and Hu, Astrophys. J., 496, 605 (1998)
#        """
#        return self.A0wh()*k**self.ns*tf.Twh(k,self.Om0,self.Ob0,self.h)**2.*self.D_plus_z(z)**2.
#
#    def DPk_bbks(self,k,z):
#        """
#		Dimensionless Matter Power Spectra Pk  as a function of k in 
#		units of [h Mpc^{-1}] and z;
#		Transfer function is taken to be BBKS 
#		Ref: Bardeen et. al., Astrophys. J., 304, 15 (1986)
#        """
#        return k**3.*self.Pk_bbks(k,z)/(2.0*np.pi**2.)
#
#    def DPk_wh(self,k,z):
#        """
#		Dimensionless Matter Power Spectra Pk  as a function of k in 
#		units of [h Mpc^{-1}] and z;
#		Transfer function is taken to be Eisenstein & Hu 
#		Ref: Eisenstein and Hu, Astrophys. J., 496, 605 (1998)
#        """
#        return k**3.*self.Pk_wh(k,z)/(2.0*np.pi**2.)


class wCDM(LCDM):
    def __init__(self,Om0,w,h=0.674,OmK=0.0,Ob0h2=0.02225,ns=0.96,sigma_8 = 0.8):
        self.Om0 = np.float64(Om0)
        self.w = np.float64(w)
        self.h = np.float64(h)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.sigma_8 = np.float64(sigma_8)
        self.Omega_r = self.Omega_r0


    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3.+self.Omega_r*(1+z)**4+self.OmK*(1+z)**2+
                       (1.-self.Om0-self.Omega_r-self.OmK)*(1.0+z)**(3.0*(1+self.w)))

    def wz(self,z):
        return z-z+self.w


class Cosmography(LCDM):
    """
    Cosmography
    """
    def __init__(self,q0,j0,s0,h=0.674,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.q0 = np.float64(q0)
        self.j0 = np.float64(j0)
        self.s0 = np.float64(s0)
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h =np.float64(h)
        self.sigma_8 = np.float64(sigma_8)

            
    def hubz(self,z):
        q0=self.q0
        j0=self.j0
        s0=self.s0
        hz=1+(1+q0)*z+(-q0**2+j0)*z**2/2+(3*q0**2+3*q0**3-4*q0*j0-3*j0-s0)*z**3/6
        return hz



class CPL(LCDM):
    """
	A Class to calculate cosmological observables for a model with CDM and dark energy
	for which equation of state w is parametrized as:
	w(a) = w0 + wa*(1-a)
	where 'a' is the scale factor and w0,wa are constants in Taylor expansion of 
	variable equation of state w(a)

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	w0 and wa: coefficients of Taylor expansion of equation of state w(a) near a=1
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
    """
    def __init__(self,Om0,w0,wa,h=0.674,OmK=0.0,Ob0h2=0.02225,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.w0 = np.float64(w0)
        self.wa = np.float64(wa)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.Omega_r = self.Omega_r0
	
    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3.+self.Omega_r*(1+z)**4+self.OmK*(1+z)**2 + (1.-self.Om0-self.Omega_r-self.OmK)*(1.+z)**(3*(1+self.w0+self.wa))*np.exp(-3.*self.wa*(z/(1.+z))))
    def wz(self,z):
        return self.w0+self.wa*z/(1.+z)

class Geos(LCDM):
    """
    arXiv:0905.4052V2
    """
    def __init__(self,Om0,w0,wa,beta,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.w0 = np.float64(w0)
        self.wa = np.float64(wa)
        self.OmK = OmK
        self.beta = np.float64(beta)
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        if abs(self.beta-0)<1e-2:
            fb=(1+z)**(-3.*(1+self.w0-self.wa/2.0*np.log(1./(1+z))))
        else:
            fb=(1+z)**(3.*(1+self.w0+self.wa/self.beta))*np.exp(3*self.wa/self.beta*(((1.+z)**(-self.beta)-1.)/self.beta))
        return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*fb)
    
    def wz(self,z):
        return self.w0+self.wa*((1.+z)**(-self.beta)-1.)/self.beta



class GCG(LCDM):
    """
	A Class to calculate cosmological observables for a model with CDM and dark energy
	for which equation of state is parametrized as that by GCG $p= -A/rho^{alpha}$:
	w(z) = -As/(As+(1-As)*(1+z)**(3*(1+alpha)))
	where 'z' is the redshift and As,alpha are model parameters.

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	As and alpha: parameters involved in eqn. of state for GCG (ref())
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
    """
    def __init__(self,Om0,As,alpha,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.As = np.float64(As)
        self.alpha = np.float64(alpha)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(self.As+(1-self.As)*(1+z)**(3*(1+self.alpha)))**(1/(1+self.alpha)))
    def wz(self,z):
        A=self.As
        a=self.alpha
        return -A/(A+(1.-A)*(1.+z)**(3.+3.*a))


class JBP(LCDM):
    """
    A Class to calculate cosmological observables for a model with CDM and dark energy
    for which equation of state w is parametrized as:
    w(a) = w0 + wa*z/(1+z)^2
    """
    def __init__(self,Om0,w0,wa,h=0.7,OmK=0.0,Ob0h2=0.02225,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.w0 = np.float64(w0)
        self.wa = np.float64(wa)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(1.+z)**(3*(1+self.w0))*np.exp(3.*self.wa*z**2/(2*(1.+z)**2)))

    def wz(self,z):
        return self.w0+self.wa*z/(1.+z)**2

class DGP(LCDM):
    """
	Physical Review D 83,043501 (2011): The Braneworld model

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	w0 and wa: coefficients of Taylor expansion of equation of state w(a) near a=1
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
    """
    def __init__(self,Om0,h=0.674,OmK=0.0,Omr0=None,Ob0h2=0.02225,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        if Omr0:
            self.Omega_r=0.0
        else:
            self.Omega_r = self.Omega_r0
	
    def hubz(self,z):
        Om_rc=(1-self.Om0-self.Omega_r)**2/4
        return np.sqrt(Om_rc)+np.sqrt(Om_rc+self.Om0*(1+z)**3+self.Omega_r*(1+z)**4)


class CGG(LCDM):
    """
	Physical Review D 83,043501 (2011) : The Chaplygin gas

	parameters are:
	Om0 : present day density parameter for total matter (baryonic + dark)
	w0 and wa: coefficients of Taylor expansion of equation of state w(a) near a=1
	Ob0 : present day density parameter for baryons (or visible matter)
	ns : spectral index for primordial power spectrum
	h : dimensionless parameter related to Hubble constant H0 = 100*h km s^-1 MPc^-1
	sigma_8 : r.m.s. mass fluctuation on 8h^-1 MPc scale
    """
    def __init__(self,Om0,k,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.k = np.float64(k)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3+self.Om0/self.k*np.sqrt(self.k**2*((1.-self.Om0)/self.Om0)**2-1.+(1.+z)**6))

class MCG(LCDM):
    """
	JCAP12(2014)043, ArXiv ePrint: 1406.7514

    """
    def __init__(self,Om0,As,B,alpha,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.As = np.float64(As)
        self.B = np.float64(B)
        self.alpha = np.float64(alpha)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        return np.sqrt(self.Om0*(1.+z)**3. + (1.-self.Om0)*(self.As+(1.-self.As)*(1.+z)**(3.*(1.+self.B)*(1.+self.alpha)))**(1./(1.+self.alpha)))
    def wz(self,z):
        A=self.As
        a=self.alpha
        B=self.B
        return B-A*(1.+B)/(A+(1.-A)*(1.+z)**(3.*(1.+B)*(1.+a)))

class PKK(LCDM):
    """
    JCAP12(2014)043, ArXiv ePrint: 1406.7514
    """
    def __init__(self,Om0,k0,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = float(Om0)
        self.k0 = float(k0)
        self.OmK = OmK
        self.Ob0h2 = float(Ob0h2)
        self.ns = float(ns)
        self.h = float(h)
        self.sigma_8 = float(sigma_8)

    def wz(self,z):
        k0=self.k0
        return -1./(1.+2.*k0**2*(1.+z)**6)
    def wxx(self,z):
        return (1.+self.wz(z))/(1.+z)
    
    @vectorize
    def qu(self,z):
        return quad(self.wxx,0,z)[0]
	
    def hubz(self,z):
        om=self.Om0
#        k0=self.k0
        fz=np.exp(3.*self.qu(z))
        return np.sqrt(om*(1.+z)**3+(1.-om)*fz)


class Pade1(LCDM):
    """
    """
    def __init__(self,Om0,w0,wa,wb,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.w0 = np.float64(w0)
        self.wa = np.float64(wa)
        self.wb = np.float64(wb)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        om=self.Om0
        w0=self.w0
        wa=self.wa
        wb=self.wb
        s1=om*(1.+z)**3
        s2=(1.-om)*(1.+z)**(3.*(1.+w0+wa+wb)/(1.+wb))
        s3=(1.+wb*z/(1.+z))**(-3.*(wa-w0*wb)/(wb*(1+wb)))
        return np.sqrt(s1+s2*s3)
    
    def wz(self,z):
        return (self.w0+(self.w0+self.wa)*z)/(1.+(1.+self.wb)*z)


class Pade2(LCDM):
    """

    """
    def __init__(self,Om0,w0,wa,wb,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.w0 = np.float64(w0)
        self.wa = np.float64(wa)
        self.wb = np.float64(wb)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
    
    def hubz(self,z):
        om=self.Om0
        w0=self.w0
        w1=self.wa
        w2=self.wb
        s1=om*(1.+z)**3
        s2=(1.-om)*(1.+z)**(3.*(w1+w2)/w2)
        s3=(1.-w2*np.log(1.+z))**(3.*(w1-w0*w2)/(w2**2))
        return np.sqrt(s1+s2*s3)
         
    def wz(self,z):
        a=1./(1.+z)
        w0=self.w0
        w1=self.wa
        w2=self.wb
        return (w0+w1*np.log(a))/(1.+w2*np.log(a))

class HDE_CA(LCDM):
    '''
    PHYSICAL REVIEW D 88, 063534 (2013)
    Cosmic age cutoff
    '''
    def __init__(self,Om0,cc,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.cc = np.float64(cc)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
      
    zz=np.linspace(0.0,5,200)
  
    def dOmegal(self,theta,z):
        OmL=theta
        cc=self.cc
        deOm=-OmL*(1.-OmL)*(3.-2*np.sqrt(OmL)/cc)/(1.+z)
        return deOm

    def Omegal(self):
        Oml=1.-self.Om0
        oml=odeint(self.dOmegal,Oml,self.zz)
        return oml
    
    def om_z(self,z):
        Oml=self.Omegal()
        f=InterpolatedUnivariateSpline(self.zz,Oml,k=5)
        return f(z)

    def wz(self,z):
        Oml=self.om_z(z)
        cc=self.cc
        w=3*np.sqrt(Oml)/3./cc-1.
        return w

    def wz_int(self,z):
        def f_int(z):
            return (1.+self.wz(z))/(1.+z)
        return quad(f_int,0,z)[0]
    
    def hubz(self,z):
        Om=self.Om0
        Oml=self.om_z(z)
        eint=np.vectorize(self.wz_int)
        E=np.sqrt(Om*(1.+z)**3+Oml*np.exp(3.*eint(z)))
        return E


class HDE_CT(LCDM):
    '''
    PHYSICAL REVIEW D 88, 063534 (2013)
    Conformal time cutoff
    '''
    def __init__(self,Om0,cc,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0 = np.float64(Om0)
        self.cc = np.float64(cc)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
      
    zz=np.linspace(0.0,5,200)
  
    def dOmegal(self,OmL,z):
        cc=self.cc
        deOm=-OmL*(1.-OmL)*(3./(1.+z)-2*np.sqrt(OmL)/cc)
        return deOm

    def Omegal(self):
        Oml=1.-self.Om0
        oml=odeint(self.dOmegal,Oml,self.zz)
        return oml
    
    def om_z(self,z):
        Oml=self.Omegal()
        f=InterpolatedUnivariateSpline(self.zz,Oml,k=5)
        return f(z)

    def wz(self,z):
        Oml=self.om_z(z)
        cc=self.cc
        w=2*np.sqrt(Oml)/cc/3.*(1.+z)-1.
        return w

    def wz_int(self,z):
        def f_int(z):
            return (1.+self.wz(z))/(1.+z)
        return quad(f_int,0,z)[0]
    
    def hubz(self,z):
        Om=self.Om0
        Oml=self.om_z(z)
        eint=np.vectorize(self.wz_int)
        E=np.sqrt(Om*(1.+z)**3+Oml*np.exp(3.*eint(z)))
        return E


class HDE_EH(LCDM):
    '''
    Eur.Phys.J.C(2016)76:588
    Event horizon cutoff
    '''
    def __init__(self,Om0,cc,h=0.7,Or0='None',OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5,0.1)):
        self.Om0 = np.float64(Om0)
        self.cc = np.float64(cc)
        self.OmK = OmK
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.Ob0h2 = np.float64(Ob0h2)    
        self.zz=np.float64(zz)
        if Or0=='None':
            self.Omega_r=self.Omega_r0
        else:
            self.Omega_r=np.float64(Or0)
        self.Ezz,self.Olzz=self.Ez_ode()
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.Ezz)
        self.f_Olz=InterpolatedUnivariateSpline(self.zz,self.Olzz)
    
    def Ez_ode(self):
        def diff_eq(y,z):
            Ez,Odz=y
            Omega_rz=self.Omega_r*(1+z)**4/Ez**2
            diff_Ez=-Odz/(1+z)*(0.5+np.sqrt(Odz)/self.cc-(Omega_rz+3)/2/Odz)*Ez
            diff_Odz=-2*Odz*(1-Odz)/(1+z)*(0.5+np.sqrt(Odz)/self.cc+Omega_rz/2/(1-Odz))
            return np.array([diff_Ez,diff_Odz])
        track=odeint(diff_eq,(1,1-self.Om0-self.Omega_r),self.zz)
        return track[:,0], track[:,1]
    
    def hubz(self,z):
        return self.f_Ez(z)
    
    def om_z(self,z):
        return self.f_Olz(z)



class RDE(LCDM):
    '''
    Eur.Phys.J.C(2016)76:588
    Ricci dark energy model
    '''
    def __init__(self,Om0,r,h=0.7,Or0='None',OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8 = 0.8,zz=np.arange(0,5,0.1)):
        self.Om0 = np.float64(Om0)
        self.r = np.float64(r)
        self.h = np.float64(h)
        if Or0=='None':
            self.Omega_r=self.Omega_r0
        else:
            self.Omega_r=np.float64(Or0)
        self.Ob0h2 = np.float64(Ob0h2)
        self.OmK = OmK
        self.ns = np.float64(ns)
        self.sigma_8 = np.float64(sigma_8)
        self.zz=np.float64(zz)

    def hubz(self,z):
        Ez1=2*self.Om0/(2-self.r)*(1+z)**3+self.Omega_r*(1+z)**4
        Ez2=(1-self.Omega_r-2*self.Om0/(2-self.r))*(1+z)**(4-2/self.r)
        return np.sqrt(Ez1+Ez2)

class IDE1(LCDM):
    """
    International Journal of Modern Physics D
    Vol. 22, No 14(2013) 1350082
    authers: Cao Shuo, Liang Nan
    model 1
    """
    def __init__(self,Om0,wx,rm,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0= np.float64(Om0)
        self.wx= np.float64(wx)
        self.rm = np.float64(rm)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
    def hubz(self,z):
        om=self.Om0
        wx=self.wx
        rm=self.rm
        return np.sqrt(wx*om/(rm+wx)*(1+z)**(3*(1-rm))+(1-wx*om/(rm+wx))*(1+z)**(3*(1+wx)))  

class RVM(LCDM):
    """
    arXiv: 1704.02136
    Running Vacuum model
    """
    def __init__(self,Om0,v,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Om0= np.float64(Om0)
        self.v= np.float64(v)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def hubz(self,z):
        om=self.Om0
        v=self.v
        return np.sqrt((om*(1+z)**(3*(1-v))+1-om-v)/(1-v))

class Rhct(LCDM):
    """
    arXiv: 1704.02136
    Running Vacuum model
    """
    def __init__(self,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8):
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.OmK = OmK
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
	
    def ang_dis_z(self,z):
#        H0=self.h*100.
        return self.D_H()*np.log(1.+z)/(1.+z)


class FT_law(LCDM):
    """
    Eur. Phys. J. C (2017) 77:502 f1CDM
    f(T)= α(−T)^b
    """
    def __init__(self,Om0,b,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5.1,0.1)):
        self.Om0 = np.float64(Om0)
        self.b = np.float64(b)
        self.Ob0h2 = np.float64(Ob0h2)
        self.OmK = OmK
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.zz = zz
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.hubzz(self.zz))
    
    @vectorize
    def hubzz(self,z):
        Om=self.Om0
        b=self.b
        a0=(1.-Om)
        def E2(E):
            B=Om*(1.+z)**3
            return E**b*a0+B-E
        fs=fsolve(E2,[1.0])  
        return fs[0]
        
    def hubz(self,z):
        return self.f_Ez(z)
    
    def weff(self,z):
        Ez=self.hubz(z)
        return (-1+self.b)*Ez**2/(self.b*(self.Om0-1)*Ez**(2*self.b)+Ez**2)

class FT_exp(LCDM):
    """
    Eur. Phys. J. C (2017) 77:502 model f2CDM
    f(T) = αT0(1−exp(−p*sqrt(T/T0)))
    """
    def __init__(self,Om0,b,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5.1,0.1)):
        self.Om0 = np.float64(Om0)
        self.b = np.float64(b)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.zz = zz
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.hubzz(self.zz))
    
    @vectorize
    def hubzz(self,z):
        Om=self.Om0
        b=self.b
        a0=(1.-Om)
        def Ez(E,z):
            yz=((E+b)*np.exp(-E/b)-b)/((b+1)*np.exp(-1/b)-b)
            return Om*(1.+z)**3+a0*yz-E**2
        fs=fsolve(Ez,[1.0],args=(z,))        
        return fs[0]
        
    def hubz(self,z):
        return self.f_Ez(z)
    
    def weff(self,z):
        Ez=self.hubz(z)
        b=self.b
        om=self.Om0
        return -((b+1)*np.exp(-1/b)-b)*((Ez**2+2*Ez*b+2*b**2)*\
                 np.exp(-Ez/b)-2*b**2)/((-om/2+0.1e1/0.2e1)*np.exp(-Ez/b)\
                 +b*((b+1)*np.exp(-1/b)-b))/((Ez+b)*np.exp(-Ez/b)-b)/2

class FT_tanh(LCDM):
    """
    Eur. Phys. J. C (2017) 77:502 f3CDM
    f(T) = α(−T)^{n}*tanh(T0/T)
    """
    def __init__(self,Om0,n,h=0.7,OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5.1,0.1)):
        self.Om0 = np.float64(Om0)
        self.n = np.float64(n)
        self.OmK = OmK
        self.Ob0h2 = np.float64(Ob0h2)
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.zz = zz
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.hubzz(self.zz))

    @vectorize
    def hubzz(self,z):
        Om=self.Om0
        n=self.n
        a0=(1.-Om)
        def Ez(E,z):
            yz=E**(2*n-2)*(2*n*np.tanh(0.1e1/E**2)*E**2-E**2*np.tanh(0.1e1/E**2)\
            +2*np.tanh(0.1e1/E**2)**2-2)/(2*np.tanh(1)**2+2*n*np.tanh(1)-np.tanh(1)-2)
            return Om*(1.+z)**3+a0*yz-E**2
        fs=fsolve(Ez,[1.0],args=(z,))        
        return fs[0]
        
    def hubz(self,z):
        return self.f_Ez(z)
    
    def weff(self,z):
        Ez=self.hubz(z)
        n=self.n
        om=self.Om0
        return (-1+np.sinh(1)*(n-0.1e1/0.2e1)*np.cosh(1))*Ez**4*\
                np.cosh(0.1e1/Ez**2)**2*(Ez**4*(n-1)*np.sinh(0.1e1/Ez**2)\
                *(n-0.1e1/0.2e1)*np.cosh(0.1e1/Ez**2)**2-2*Ez**2*(n-0.5e1/0.4e1)\
                *np.cosh(0.1e1/Ez**2)-2*np.sinh(0.1e1/Ez**2))/(-2*np.cosh(1)**2*(om-1)\
                *np.cosh(0.1e1/Ez**2)*(n-0.3e1/0.4e1)*Ez**(2+2*n)+np.sinh(0.1e1/Ez**2)*\
                np.cosh(1)**2*(om-1)*(n-0.1e1/0.2e1)*n*np.cosh(0.1e1/Ez**2)**2*Ez**(4+2*n)\
                +(-1+np.sinh(1)*(n-0.1e1/0.2e1)*np.cosh(1))*Ez**6*np.cosh(0.1e1/Ez**2)**3\
                -2*np.cosh(1)**2*np.sinh(0.1e1/Ez**2)*Ez**(2*n)*(om-1))/(-1+np.sinh(0.1e1/Ez**2)\
                *Ez**2*(n-0.1e1/0.2e1)*np.cosh(0.1e1/Ez**2))

class fR_power(LCDM):
    def __init__(self,Om0,nn,bb,h=0.7,Or0='None',OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5,0.1),inter=True):
        self.Om0 = np.float64(Om0)
        self.nn = np.float64(nn)
        self.bb = np.float64(bb)
        self.OmK = OmK
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.Ob0h2 = np.float64(Ob0h2)    
        self.zz=np.float64(zz)
        if Or0=='None':
            self.Omega_r=self.Omega_r0
        else:
            self.Omega_r=np.float64(Or0)
        self.Ezz=np.sqrt(self.H2z(self.zz)/self.h**2/1e4)
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.Ezz)
        self.inter=inter
    
    
    def hubz(self,z):
        if self.inter==True:
            return self.f_Ez(z)
        else:
            return np.sqrt(self.H2z(z))
    
    
    def fR_function(self,R):
        f=R-self.bb/R**self.nn
        fR=1+self.bb/R**self.nn*self.nn/R
        fRR=-self.bb/R**self.nn*self.nn**2/R**2-self.bb/R**self.nn*self.nn/R**2
        return f,fR,fRR
    
    @vectorize
    def solve_R(self,z):
        H0=self.h*1e2
        Matter=self.Om0*(1+z)**3+self.Omega_r*(1+z)**4
        def RR(R,z):
            f,fR,fRR=self.fR_function(R)
            return R*fR-2*f+3*H0**2*Matter
        fs=fsolve(RR,[2.0],args=(z,))
        return fs[0]
    
    def H2z(self,z):
        R=self.solve_R(z)
        f,fR,fRR=self.fR_function(R)
        H2u=3*f-R*fR
        H2d=6*fR*(1-1.5*fRR*(R*fR-2*f)/fR/(R*fRR-fR))**2
        return H2u/H2d

class fR_power2(LCDM):
    def __init__(self,Om0,nn,h=0.7,Or0='None',OmK=0.0,Ob0h2=0.02236,ns=0.96,sigma_8=0.8,zz=np.arange(0,5,0.1),inter=True):
        self.Om0 = np.float64(Om0)
        self.nn = np.float64(nn)
        self.OmK = OmK
        self.ns = np.float64(ns)
        self.h = np.float64(h)
        self.sigma_8 = np.float64(sigma_8)
        self.Ob0h2 = np.float64(Ob0h2)    
        self.zz=np.float64(zz)
        if Or0=='None':
            self.Omega_r=self.Omega_r0
        else:
            self.Omega_r=np.float64(Or0)
        self.Ezz=np.sqrt(self.H2z(self.zz))
        self.f_Ez=InterpolatedUnivariateSpline(self.zz,self.Ezz)
        self.inter = inter

    @property
    def Omega_r0(self):
        T_CMB=2.7255
        z_eq=2.5e4*self.Om0*self.h**2*(T_CMB/2.7)**(-4)
        return self.Om0/(1.0+z_eq)    
    
    def hubz(self,z):
        if self.inter==True:
            return self.f_Ez(z)
        else:
            return np.sqrt(self.H2z(z))
    
    def fR_function(self,R):
#        H0=self.h*1e2
        H0=1
        def fRz0(R):
            return ((self.nn+1)*R-3*H0**2*self.Om0*self.nn)**2*((self.nn+1)*R
                    -0.3e1/0.2e1*H0**2*self.Om0*self.nn)*\
                    ((self.nn+1)*R+3*H0**2*self.Om0*(self.nn+3))*R/((self.nn+1)**2*R**2\
                     -0.9e1/0.4e1*self.nn*H0**2*self.Om0*(self.nn+1)*R-\
                     0.9e1/0.4e1*self.nn*H0**4*self.Om0**2*(self.nn+3))**2/H0**2/12-1
        R0=fsolve(fRz0,[10.0])[0]
        bb=R0**self.nn*(R0-3*H0**2*self.Om0)/(self.nn+2)
        f=R-bb/R**self.nn
        fR=1+bb/R**self.nn*self.nn/R
        fRR=-bb/R**self.nn*self.nn**2/R**2-bb/R**self.nn*self.nn/R**2
        return f,fR,fRR
    
    @vectorize
    def solve_R(self,z):
#        H0=self.h*1e2
        H0=1
        Matter=self.Om0*(1+z)**3
        def RR(R,z):
            f,fR,fRR=self.fR_function(R)
            return R*fR-2*f+3*H0**2*Matter
        fs=fsolve(RR,[20.0],args=(z,))
        return fs[0]
    
    def H2z(self,z):
#        H0=self.h*1e2
#        H0=1
        R=self.solve_R(z)
        f,fR,fRR=self.fR_function(R)
        H2u=3*f-R*fR
        H2d=6*fR*(1-1.5*fRR*(R*fR-2*f)/fR/(R*fRR-fR))**2
        return H2u/H2d
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 02 15:20:26 2017

@author: jingzhao
"""
__package__ = 'qcosmc'
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splint,splev
from scipy.misc import derivative
from scipy import integrate
#from time import clock
#from sklearn.gaussian_process import GaussianProcessRegressor
from .FigStyle import qplt
dd=np.dtype([('name',np.str_,16),('num',np.int),('lower',np.float64),('upper',np.float64)])



def get_errors(FF):
    if FF.ndim == 2:
        cov=np.linalg.inv(FF)
        err=np.sqrt(cov.diagonal())
    elif FF.ndim==3:
        n=FF.shape[0]
        err=np.zeros((FF.shape[1],FF.shape[0]))
        for i in range(n):
            cov=np.linalg.inv(FF[i])
            err[:,i]=np.sqrt(cov.diagonal())
    else:
        raise ValueError("Matrix.ndim should be 2 or 3.")
    return err

def Fisher2Fisher(z,equa,param,FF):
    '''
    Calculate a new Fisher marix by using the transformation matrix

    Parameters
    ----------
    z : float or array
        redshift.
    equa : list
        The list of function names, for example [fsig8,DA,Hz]
    param : list
        The list of parameter values, for example [70,0.3,-1,0] are the values of H0, Omega_m0, w0, wa, respectively, which are the parameters of fsig8, DA, Hz.
    FF : Matrix
       old Fisher.

    Returns
    -------
    Fab2 : Matrix
        new Fisher.

    '''
    if type(z)==list: z=np.asarray(z)
    if type(z) != np.ndarray:
        MM=transformation_matrix(z,equa,param)
        Fab2=MM.T@FF@MM
    else:
        Fab2=0
        for i,Fs in enumerate(FF):
            MM=transformation_matrix(z[i],equa,param)
            Fab2+=MM.T@Fs@MM
    return Fab2

def fix_param_Fisher(Fisher,var):
    return del_diag(Fisher, var)

def add_priors(Fisher, var, error):
    '''
    add a prior of parameter and get a new Fisher

    Parameters
    ----------
    Fisher : Matrix
        
    var : int
        the i-th variable.
    error : float
        the uncertainty of corresponding variable

    Returns
    -------
    Fisher : TYPE
        DESCRIPTION.

    '''
    Fisher[var,var]=Fisher[var,var]+1/error**2
    return Fisher

def marginalization(Fisher,var):
    '''
    marginalize over a variable of the given Fisher matrix and return a new Fisher matrix

    Parameters
    ----------
    Fisher : Matrix
        The Fisher matrix to be marginalized
    var : int
        the i-th variable.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    cov=Fisher.I
    New_cov=del_diag(cov,var)
    return New_cov.I

def Fisher(z,func,params,df):
    '''
    
    Parameters
    ----------
    z : float or numpy.ndarray
        redshift
    func : function
        the function to be derivated
    params : list
        a list of parameters list to function excepting for redshift
    df : float
        the uncertainty of function value

    Returns
    -------
    a Fisher matrix
    '''
    if type(z) != np.ndarray:
        return Fisherz(z,func,params,df)
    else:
        if len(z)!= df.size:
            raise ValueError("The type of 'df' must be array if 'z' you input is array.")
        FF=0
        for i,Fs in enumerate(z):
            FF+=Fisherz(z[i],func,params,df[i])
        return FF

def Fisherz(z,func,params,df):
    
    '''
    Parameters
    ----------
    z : float
        redshift
    func : function
        the function to be derivated
    params : list
        a list of parameters list to function excepting for redshift
    df : float
        the uncertainty of function value

    Returns
    -------
    a Fisher matrix

    '''
    n=len(params)
    FF=np.zeros((n,n))
    for i in range(n):
        F1=partial_derivative(func,i,[*params,z])
        for j in range(i,n):
            F2=partial_derivative(func,j,[*params,z])
            FF[i,j]=F1*F2
    FF += FF.T - np.diag(FF.diagonal())
    return np.matrix(FF)/df**2
 
def transformation_matrix(z,equa,param):
    '''
    Calculate the transformation matrix

    Parameters
    ----------
    z : float
        redshift
    equa : list
        The list of function names, for example [fsig8,DA,Hz]
    param : list
        The list of parameter values, for example [70,0.3,-1,0] are the values of H0, Omega_m0, w0, wa, respectively, which are the parameters of fsig8, DA, Hz.

    Returns
    -------
    transformation matrix
    
    for example
    [dfsig8/dH0, dfsig8/dOmega_m0, dfsig8/dw0, dfsig8/dwa]
    [dDA/dH0, dDA/dOmega_m0, dDA/dw0, dDA/dwa]
    [dHz/dH0, dHz/dOmega_m0, dHz/dw0, dHz/dwa]
    
    '''
    en=len(equa)
    pn=len(param)
    M=np.zeros((en,pn))
    for i,fun in enumerate(equa):
        for j, par in enumerate(param):
            M[i,j]=partial_derivative(fun,j,[*param,z])
    return np.matrix(M)

def del_diag(matrix,i):
    '''
    Delete i-th row and column from the matrix.
    del_diag(a,1) or del_diag(a,[1,2])
    
    Parameters
    ----------
    matrix : 2-D array
    i : integer or list of integer

    Returns
    -------
    2-D array
        a new matrix.
    '''
    return np.delete(np.delete(matrix,i,1),i,0)

def partial_derivative(func, var=0, point=[],dx=1e-6):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = dx)

def mean(mu,sig):
    mubar=np.sum(mu/sig**2)/np.sum(1./sig**2)
    sigbar=np.sqrt(1./np.sum(1./sig**2))
    return mubar,sigbar

def fen_bins(z,f,f_s,bins=[0,5,1]):
    b=np.arange(bins[0],bins[1]+bins[2],bins[2])
    zb=[]
    fb=[]
    fb_s=[]
    for i in range(b.size-1):
        n= [j for j in range(z.size) if b[i]<= z[j] < b[i+1]]
        zb.append(z[n].mean())
        ff,ff_s=mean(f[n],f_s[n])
        fb.append(ff)
        fb_s.append(ff_s)
    return zb,fb,fb_s

#def GP(zs,hs,hs_sig,cXstar,**kwargs):
#    zz=np.atleast_2d(zs).T
#    gp = GaussianProcessRegressor(alpha=(hs_sig /hs) ** 2,
#                              n_restarts_optimizer=10,normalize_y=True,**kwargs)
#    gp.fit(zz,hs)
#    zstar=np.linspace(cXstar[0],cXstar[1],cXstar[2])
#    zst=np.atleast_2d(zstar).T
#    H_pred, H_sigma = gp.predict(zst, return_std=True)
#    return zstar, H_pred, H_sigma

def savetxt(filename,aa,**kwargs):
    rec=np.transpose(aa)
    np.savetxt(filename,rec,fmt='%f',**kwargs)

def isnan(xx):
    n=[]
    for i,x in enumerate(xx):
        if np.isnan(x):
            n.append(i)
    return n


def del_nan(*args):
    n=[]
    for xx in args:
        n+=isnan(xx)
    if n:
        n=list(set(n))
    temp = tuple(np.delete(arg,n) for arg in args)
    return temp if len(temp) > 1 else temp[0]


def mu_to_Dl(mu,mu_sig):
    def mu_D(mu):
        return 10.0**((mu-25.0)/5.0)
    Dl_sig=abs(derivative(mu_D,mu)*mu_sig)
    return mu_D(mu),Dl_sig

def calibration_sn(fgs,sn,error):
    fgsz=fgs[0,:]
    fn=len(fgsz)
    dl=[]
    dl_sig=[]
    fz=f=f_s=[]
    for i in range(fn):
        c1=np.where(abs(fgsz[i]-sn[0,:])<=error)
        if list(c1[0]):
            mub=np.sum(sn[1,:][c1]/sn[2,:][c1]**2)/np.sum(1/sn[2,:][c1]**2)
            mub_sig=np.sqrt(1.0/np.sum(1/sn[2,:][c1]**2))
            Dl,Dl_sig=mu_to_Dl(mub,mub_sig)
            dl=np.append(dl,Dl)
            dl_sig=np.append(dl_sig,Dl_sig)
            fz=np.append(fz,fgsz[i])
            f=np.append(f,fgs[1,i])
            f_s=np.append(f_s,fgs[2,i])
    return fz,f,f_s,dl,dl_sig

def calibration(fgsz,sn,error):
    fn=len(fgsz)
    mub=[]
    mub_sig=[]
    zn=[]
    for i in range(fn):
        c1=np.where(abs(fgsz[i]-sn[0,:])<=error)
        if list(c1[0]):
            mub.append(np.sum(sn[1,:][c1]/sn[2,:][c1]**2)/np.sum(1/sn[2,:][c1]**2))
            mub_sig.append(np.sqrt(1.0/np.sum(1/sn[2,:][c1]**2)))
            zn.append(i)
    return zn,np.asarray(mub),np.asarray(mub_sig)

def redshift_match(z,target_z,error):
    n=len(z)
    zn=[]
    tar_zn=[]
    for i in range(n):
        c1=np.where(abs(z[i]-target_z)<=error)
        if list(c1[0]):
            zn.append(i)
            tn=list(np.random.choice(c1[0],1))
            tar_zn.append(tn[0])
    return zn,tar_zn

def redshift_match_SGL(zl,zs,qz,err=5e-3):
    sgln=[]
    qzln=[]
    qzsn=[]
    nn=len(zl)
    for i in range(nn):
        nzl=np.where(abs(zl[i]-qz)<=err)[0]
        nzs=np.where(abs(zs[i]-qz)<=err)[0]
        if len(nzl)>=1: nzl=np.random.choice(nzl)
        if len(nzs)>=1: nzs=np.random.choice(nzs)
        if nzl.any() and nzs.any():
            sgln.append(i)
            qzln.append(nzl)
            qzsn.append(nzs)
    return sgln,qzln,qzsn
    

def random_fun(funn,xmin,xmax,number,bin_with,fig=False,**kwargs):
    '''
    函数功能：根据概率密度函数生成随机数。
    funn : 概率密度函数，或者是散点的x,y列表：[xx,xy]
    xmin : 随机数的最小值
    xmax : 随机数的最大值
    number: 生成随机数数量。
    bin_with: 步长，步长越小生成的越精细，但耗费时间也越长。建议是(xmax-xmin)/20
    fig : 如果fig=True 则会画出示意图（包括直方图和概率密度函数）
    '''
    if funn.__class__.__name__=='list':
        fun=lambda x: splev(x,splrep(funn[0],funn[1]))
        nor=True
    else:
        fun=funn
        nor=False
    zz=np.arange(xmin,xmax+bin_with,bin_with)
    N=len(zz)
    Ngw=np.zeros(N-1)
    bins=np.zeros([N-1,2])
    addn=0
    for i in range(N-1):
        bins[i:]=[zz[i],zz[i+1]]
        Ngw[i]=integrate.quad(fun,zz[i],zz[i+1])[0]
        
    ratio=Ngw/np.sum(Ngw)
    true_n=number-1
    while true_n<number:
        fb=list(map(int,map(round,(number+addn)*ratio)))
        true_n=sum(fb)
        addn=addn+1
#    print('The Number of simulated data is %s'%true_n)
    zzn=[]
    for i in range(len(fb)):
        zn=np.random.uniform(bins[i,0],bins[i,1],fb[i])
        zzn=np.append(zzn,zn)
    ans=np.random.choice(zzn,number,replace=False)
    if fig:
        plt.figure()
#        plt.yticks([])
        bbb=plt.hist(ans,bins=number//10,density=True,color='g',alpha=0.5,edgecolor='k')
        zs=np.arange(xmin,xmax+bin_with,bin_with)
        if nor:
            qplt(zs,fun(zs),lw=2,**kwargs)
        else:
            qplt(zs,fun(zs)/np.max(fun(zs))*np.max(bbb[0]),lw=2,**kwargs)
#        qplt(zs,fun(zs)/np.max(fun(zs)),lw=2,**kwargs)
    return np.sort(ans)




#@timer
#def gedian2(pars,chi2,fig_n='',chain_n=''):
#    chains_dir='./chains/'
#    outdir='./results/'
#    params_all=np.zeros(0,dtype=dd)
#    for i in range(len(pars)):
#        params_all=np.append(params_all,np.array([tuple(pars[i])],dtype=dd))
#   
#    ll=np.linspace(params_all['lower'][0],params_all['upper'][0],params_all['num'][0])
#    kk=np.linspace(params_all['lower'][1],params_all['upper'][1],params_all['num'][1])
#    ans=[chi2([i,j]) for i in ll for j in kk]
#    kaf00=np.reshape(ans,(params_all['num'][0],params_all['num'][1])).T
#    
#    if chain_n:
#        np.save(chains_dir+chain_n+"_g.npy",(kaf00,ll,kk,params_all['name'][0],params_all['name'][1]))
#    
#    plot_2D(kaf00,ll,kk,params_all['name'][0],params_all['name'][1])
#    if fig_n:
#        plt.savefig(outdir+fig_n+'.pdf')
#    return ll,kk,kaf00

def GP_int(rec):
    z,g,sig=rec
    n=len(z)
    tck = splrep(z,g)
    gint=np.zeros(n)
    for i in range(n):
        gint[i]=splint(0,z[i],tck)
    tck_s=splrep(z, g+sig)
    gint_s=np.zeros(n)
    for i in range(n):
        gint_s[i]=splint(0,z[i],tck_s)
    return z,gint,gint_s-gint


def GP_plot(z,Da,sig,rec,xlabel,ylabel,fig_name=None,text_style=False,xlim=None,ylim=None,label='',data_label=''):
    if text_style:
        plt.rc('text', usetex=True)
    fc='#b0a4e2'
    plt.errorbar(z, Da, sig, fmt='o',label=data_label)
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                         facecolor=fc,lw=0,alpha=1,label=label)
    plt.fill_between(rec[:, 0], rec[:, 1] + 2*rec[:, 2], rec[:, 1] - 2*rec[:, 2],
                         facecolor=fc,lw=0,alpha=0.5)
    plt.plot(rec[:, 0], rec[:, 1],'-',color='#c83c23')  
    plt.xlabel('$'+xlabel+'$')
    plt.ylabel('$'+ylabel+'$')
    plt.xlim(xlim)
    plt.ylim(ylim)
    #ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    if fig_name :
        plt.savefig(fig_name)

def GP_p(rec,label=''):
    fc='#348ABD'
    plt.fill_between(rec[:, 0], rec[:, 1] + rec[:, 2], rec[:, 1] - rec[:, 2],
                         facecolor=fc,lw=0,alpha=0.6,label=label)
    plt.fill_between(rec[:, 0], rec[:, 1] + 2*rec[:, 2], rec[:, 1] - 2*rec[:, 2],
                         facecolor=fc,lw=0,alpha=0.3)
    plt.plot(rec[:, 0], rec[:, 1],'-',color='#348ABD')      

#-------------------------------------------------------------------------------------------
def reconstruction(re_func,p,pu,pd,z):
    n= len(z)
    pn=len(p)
    dfdp=np.zeros((n,pn))
    dfdpmax=np.zeros((n,pn))
    dfdpmin=np.zeros((n,pn))
    
    re_fun_u = np.zeros(n)
    re_fun_d = np.zeros(n)
    re_fun_f = np.zeros(n)
    
    def partial_derivative(func, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)   
    pp=list(p)
    zz=list(z)
       
    for i in range(n):
        pam=pp+[zz[i]]
        for k in range(pn):
            dfdp[i][k]=partial_derivative(re_func,k,pam) 
            dfdpmax[i][k]=max(dfdp[i][k]*pu[k],-dfdp[i][k]*pd[k])**2
            dfdpmin[i][k]=min(dfdp[i][k]*pu[k],-dfdp[i][k]*pd[k])**2
        re_fun_u[i]=np.sqrt(sum(dfdpmax[i]))
        re_fun_d[i]=np.sqrt(sum(dfdpmin[i]))
        re_fun_f[i]=re_func(*pam) 
    return re_fun_f,re_fun_f+re_fun_u, re_fun_f-re_fun_d


def simp_err(re_func,p,pu):
    para=np.asarray(p)
    para_err=np.asarray(pu)
    n=para.shape[1]
    f=np.zeros(n)
    gam_sig=np.zeros(n)
    
    for i in range(n):
        pp=list(para[:,i])
        perr=list(para_err[:,i])
        f[i],gam_sig[i]=error_transfer(re_func,pp,perr)
    return f,gam_sig

def error_transfer(re_func,p,pu):
    pn=len(p)
    dfdp=np.zeros(pn)
    dfdp_err=np.zeros(pn)
      
    def partial_derivative(func, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)   
    pp=list(p)

    for k in range(pn):
        dfdp[k]=partial_derivative(re_func,k,pp) 
        dfdp_err[k]=(dfdp[k]*pu[k])**2
    re_fun_err=np.sqrt(sum(dfdp_err))
    re_fun_f=re_func(*pp) 
    return re_fun_f,re_fun_err

def err_ts(func,f,err):
    ref=func(f)
    re_err=abs(derivative(func,f,dx=1e-6)*err)
    return ref,re_err
    

def model_recon(Chains_name,re_func,re_z):
    savefile_name='./chains/'+Chains_name+'.npy'
    samples,theta_name,theta_fit,theta_fact,minkaf,data_num=np.load(savefile_name)
    p=theta_fact[0]
    pu=theta_fact[1]
    pd=theta_fact[2]
    f_mc,u_mc,d_mc=reconstruction(re_func,p,pu,pd,re_z)
    return f_mc,u_mc,d_mc
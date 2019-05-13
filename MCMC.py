# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:02:38 2017

@author: jingzhao
"""

import numpy as np
import emcee
import os
import sys
import scipy.optimize as opt
from time import strftime,localtime
from .Decorator import timer

dd=np.dtype([('name',np.str_,16),('fit',np.float64),('lower',np.float64),('upper',np.float64)])

class MCMC_class(object):
    def __init__(self,parameters,chi2,Chains_name,data_num=0):
        self.chi2=chi2
        self.Chains_name= Chains_name
        self.data_num = data_num
        self.params_all=np.zeros(0,dtype=dd)
        for i in range(len(parameters)):
            self.params_all=np.append(self.params_all,np.array([tuple(parameters[i])],dtype=dd))

        self.n=len(self.params_all)
        self.theta_fit=np.zeros(self.n)
        self.theta_fact=np.zeros(self.n)

    def _lnprior(self,x):
        i=0
        while i<len(self.params_all):
            if self.params_all['lower'][i]<=x[i]<=self.params_all['upper'][i]:
                s=0.0
            else:
                s=-np.inf
                break
            i=i+1
        return s
#    

    def _lik(self,theta):
    	return np.exp(-self.chi2(theta)/2.)
    
    def _lnlike(self,theta):
        return np.log(self._lik(theta))
#    	
    def _lnprob(self,theta):
        lp = self._lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp+self._lnlike(theta)
#    
    def _ff(self,theta):
        return -2.0*self._lnprob(theta)
#    
    def _lnp(self,theta):
        return self._lnprob(theta)
    def _check_err(self,gv,fv):
        if gv==fv:
            while(True):
                print('你的程序可能有问题。')
                print('%s=%s'%(self.chi2.__name__,self.chi2(self.params_all['fit'])))
                print('中断，按\'b\'，\n继续，按\'c\'。')
                print('如果想要一些帮助，按\'h\'。')
                p1=input()
                if p1=='b':
                    sys.exit(0)
                elif p1=='c':
                    return
                elif p1=='h':
                    print('出现这个问题的原因通常是X^2_min值太大或者太小')
                    print('所以检查你的参数输入范围和中心值是否合理')
                    print('如果没问题，检查X^2值，看是否和你用的数据点查了好几个数量级，就下面这个值')
                    print('%s=%s'%(self.chi2.__name__,self.chi2(self.params_all['fit'])))
                    print('如果差了好几个数量级，那就是你chi2程序有问题。')
                    print('中断程序按\'b\'，继续跑按\'c\'')
                    p2=input()
                    if p2=='b':
                        sys.exit()
                    elif p2=='c':
                        break
                    else:
                        return
                else:
                    print('输入错误，重新输入')

    def Rc(self,chains,nwalkers):
        R_c = np.linspace(0, 1, chains.shape[-1])
        print('-'*40)
        for i in range(chains.shape[-1]):
            para = chains[:,i]
            para_reshape = para.reshape((-1,nwalkers))
            para_reshape = para_reshape[para_reshape.shape[0]//2:para_reshape.shape[0],:]   # the second half of each chain is used to check the convergence state
            walker_mean = np.mean(para_reshape, axis=0, keepdims=True) # mean of each walker
            var_mean = np.var(walker_mean)                             # variance between each walker
            walker_var = np.var(para_reshape, axis=0, keepdims=True)   # variance of each walker
            mean_var = np.mean(walker_var)
            
           # sample from one walker ==> one chain
           # For multiple (nwalkers) chains  the code computes the Gelman and Rubin "R statistic"
           # Please See Page 38 of "eprint arXiv:0712.3028" for the definitions of "R statistic"
           
            R_c[i] = (mean_var*(1.0-2.0/para_reshape.shape[0])+var_mean*(1.0+1.0/nwalkers))/mean_var 
            print('R[%s]-1='%i+'%s'%abs(R_c[i]-1))
        return R_c

    @timer
    def MCMC(self,steps=500,nwalkers=100):
        print ('\n'+'=======================================================')
        print (strftime("%Y-%m-%d %H:%M:%S",localtime())+'\n')
        result = opt.minimize(self._ff,self.params_all['fit'],method='Nelder-Mead')
        self.theta_fit= result['x']
        for i in range(self.n):
            print("""The best-fit value of {0}={1}""".format(self.params_all['name'][i],self.theta_fit[i]))
        
        self._check_err(self.params_all['fit'][0],self.theta_fit[0])
        
        # Set up the sampler.
        ndim= self.n
        pos = [result['x'] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnp)
        #print pos
        # Clear and run the production chain.
        print("Running MCMC...")
        sampler.run_mcmc(pos, steps)
        print("Done.")
        
        chains_dir='./chains/'
        if not os.path.exists(chains_dir):
            os.makedirs(chains_dir)
        # Make the triangle plot.
        savefile_name='./chains/'+self.Chains_name+'.npy'
        burnin = 50
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        self.Rc(samples,nwalkers)
        vv=lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        self.theta_fact= vv(np.percentile(samples, [16, 50, 84],axis=0))
#        self.theta_fact=0.0
        self.minkaf=self.chi2(self.theta_fit)
        self.samples=samples
        ranges=list(zip(self.params_all['lower'],self.params_all['upper']))
        
        np.save(savefile_name,(samples,self.params_all['name'],self.theta_fit,self.theta_fact,self.minkaf,self.data_num,ranges))
        print('\nChains name is "%s".'%self.Chains_name+'  data number:%s'%self.data_num)

    def check(self,*arg):
        if arg:
            print('%s=%s'%(self.chi2.__name__,self.chi2(arg)))
        else:
            print('%s=%s'%(self.chi2.__name__,self.chi2(self.params_all['fit'])))
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:02:38 2017

@author: jingzhao
"""
__package__ = 'qcosmc'
import numpy as np
import emcee
import os
import sys
import scipy.optimize as opt
from time import strftime,localtime
import matplotlib.pyplot as plt
from getdist import MCSamples,loadMCSamples,plots
from getdist.gaussian_mixtures import GaussianND
from .FigStyle import qstyle
from .Decorator import timer
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"

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

    def savefile(self,sampler,nwalkers):
        chains_dir='./chains/'
        if not os.path.exists(chains_dir):
            os.makedirs(chains_dir)
        
        savefile_name='./chains/'+self.Chains_name+'.npy'
        burnin = 100
        samples = sampler.chain[:, burnin:, :].reshape((-1, self.n))
        self.Rc(samples,nwalkers)
        vv=lambda v: (v[1], v[2]-v[1], v[1]-v[0])
        self.theta_fact= vv(np.percentile(samples, [16, 50, 84],axis=0))
        self.minkaf=self.chi2(self.theta_fit)
        self.samples=samples
        ranges=list(zip(self.params_all['lower'],self.params_all['upper']))
        
        np.save(savefile_name,(samples,self.params_all['name'],self.theta_fit,self.theta_fact,self.minkaf,self.data_num,ranges))
        print('\nChains name is "%s".'%self.Chains_name+'  data number:%s'%self.data_num)

    @timer
    def MCMC(self,steps=1000,nwalkers=100,nc=1e-4):
        print ('\n'+'=======================================================')
        print (strftime("%Y-%m-%d %H:%M:%S",localtime())+'\n')
        result = opt.minimize(self._ff,self.params_all['fit'],method='Nelder-Mead')
        self.theta_fit= result['x']
        for i in range(self.n):
            print("""The best-fit value of {0}={1}""".format(self.params_all['name'][i],self.theta_fit[i]))
        
        self._check_err(self.params_all['fit'][0],self.theta_fit[0])
        
        # Set up the sampler.
        ndim= self.n
        pos = [result['x'] + nc*np.random.randn(ndim) for i in range(nwalkers)]
        # print(pos)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnp)
        #print pos
        # Clear and run the production chain.
        print("Running MCMC...")
        try:
            sampler.run_mcmc(pos, steps, progress=True)
        except ValueError:
            print(pos)
            raise ValueError("Probability function returned NaN")
        print("Done.")
        self.savefile(sampler,nwalkers)
        return sampler
    

    def MCMC_mul(self,steps=1000,nwalkers=100,nc=1e-4):
        print ('\n'+'=======================================================')
        print (strftime("%Y-%m-%d %H:%M:%S",localtime())+'\n')
        result = opt.minimize(self._ff,self.params_all['fit'],method='Nelder-Mead')
        self.theta_fit= result['x']
        for i in range(self.n):
            print("""The best-fit value of {0}={1}""".format(self.params_all['name'][i],self.theta_fit[i]))
        
        self._check_err(self.params_all['fit'][0],self.theta_fit[0])
        
        # Set up the sampler.
        ndim= self.n
        pos = [result['x'] + nc*np.random.randn(ndim) for i in range(nwalkers)]
        # print("Running MCMC...")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnp)
            # sampler.run_mcmc(pos, steps, progress=True)
        # print("Done.")
        # self.savefile(sampler,nwalkers)
        return sampler, pos

    def check(self,*arg):
        if arg:
            print('%s=%s'%(self.chi2.__name__,self.chi2(arg)))
        else:
            print('%s=%s'%(self.chi2.__name__,self.chi2(self.params_all['fit'])))


qstyle(0)
colors=['#348ABD', '#A60628', '#467821', '#7A68A6',  '#E24A33' ,'#ffb3a6', '#188487']
lss=['-','--','-.',':']
outdir='./results/'

class MCplot(object):
    def __init__(self,Chains,new_name=None,ignore_rows=0.1):
        self.root=list(np.asarray(Chains)[:,0])
        self.lengend=list(np.asarray(Chains)[:,1])
        self.aic_g=True
        self.Samp=[]
        self._n = len(Chains)
        self.minkaf=np.zeros(self._n)
        self.data_num=np.zeros(self._n)
#        self.theta_fit=np.zeros(self._n)
#        self.theta_fact=np.zeros(self._n)
        for i in range(self._n):
            savefile_name='./chains/'+self.root[i]+'.npy'
            self.samples,self.theta_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges=np.load(savefile_name, allow_pickle=True)
            if new_name:
                self.theta_name=new_name
            self.label_name=[x.replace('H_0','H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]') for x in self.theta_name]
            self.Samp.append(MCSamples(samples=self.samples,names = self.theta_name, labels = self.label_name,ranges=ranges,settings={'ignore_rows':ignore_rows}))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
    
    def rename(self,new_name):
        for i in range(self._n):
            savefile_name='./chains/'+self.root[i]+'.npy'
            samples,self.theta_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges=np.load(savefile_name, allow_pickle=True)
            np.save(savefile_name,(samples,new_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
    
    def plot1D(self,n,colorn=0,width_inch=8,**kwargs):
        g = plots.getSinglePlotter(width_inch=width_inch)
        g.plot_1d(self.Samp,self.param_names[n-1],ls=lss,colors=colors[colorn:colorn+self._n],lws=[1.5]*self._n,**kwargs)
        # g.settings.figure_legend_frame = False
        ax=plt.gca()
        if all(self.lengend):
            leg = ax.legend(self.lengend,loc=1,fontsize=16,frameon=False)
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'],lw=1.5)
#        if 'x_bands' in kwargs:
#            g.add_x_bands(0,0.01)
        if 'xaxis' in kwargs:
            ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['xaxis']))
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        # plt.tight_layout()
        if 'name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['name']))
        else:
            g.export(os.path.join(outdir+''.join(self.root)+self.param_names[n-1].replace('\\','')+'_1D.pdf'))
        return g
    
    
    def plot2D(self,pp,colorn=0,contour_num=2,width_inch=8,**kwargs):
        g = plots.getSinglePlotter(width_inch=width_inch,ratio=1)
        g.settings.num_plot_contours = contour_num
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 18
        g.settings.legend_frame = False
        g.plot_2d(self.Samp,self.param_names[pp[0]-1],self.param_names[pp[1]-1],filled=True,**kwargs)
        if 'x_locator' in kwargs:
            ax=g.get_axes()
            ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['x_locator']))
            del kwargs['x_locator']
        if 'y_locator' in kwargs:
            ax=g.get_axes()
            ax.yaxis.set_major_locator(plt.MultipleLocator(kwargs['y_locator']))
            del kwargs['y_locator']
        if 'lims' in kwargs:
            [xmin, xmax, ymin, ymax]=kwargs['lims']
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            del kwargs['lims']
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'],lw=1.5)
        if all(self.lengend):
            kwarg=kwargs.copy()
            if 'name' in kwarg: del kwarg['name']
            g.add_legend(self.lengend,colored_text=True, fontsize=16,**kwarg)
        # plt.tight_layout()
        if 'name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['name']))
        else:
            g.export(os.path.join(outdir,''.join(self.root)+'_2D.pdf'))
        return g


    def plot3D(self,pp,colorn=None,contour_num=2,**kwargs):
        if colorn:
            colorss=colors[colorn-1:colorn-1+self._n]
        else:
            colorss=None
            
        if pp==0:
            t_name=self.param_names
        else:
            t_name=[]
            for i in pp:
                t_name.append(self.param_names[i-1])
        g = plots.get_subplot_plotter(width_inch=9)
        g.settings.num_plot_contours = contour_num
        g.settings.legend_fontsize = 20
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 18
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.8
        if all(self.lengend):
            print(t_name)
            g.triangle_plot(self.Samp,t_name,filled_compare=True,legend_labels=self.lengend,contour_colors=colorss,legend_loc='upper right',**kwargs)
        else:
            g.triangle_plot(self.Samp,t_name,filled_compare=True,contour_colors=colorss,**kwargs)
        if 'xlim' in kwargs:
            for xi in kwargs['xlim']:
                for ax in g.subplots[:,xi[0]-1]:
                    if ax is not None:
                        ax.set_xlim(xi[1][0],xi[1][1])
        if 'tline' in kwargs:
            for ax in g.subplots[:,0]:
                ax.axvline(kwargs['tline'], color='k', ls='--',alpha=0.5)
#        plt.tight_layout()
        if 'ax_range' in kwargs:
            for axi in kwargs['ax_range']:
                g.subplots[-1,axi[0]-1].xaxis.set_major_locator(plt.MultipleLocator(axi[1]))
                if axi[0]-1>0:
                    g.subplots[axi[0]-1,0].yaxis.set_major_locator(plt.MultipleLocator(axi[1]))
        if 'name' in kwargs:
            g.export(os.path.join(outdir,'%s.pdf'%kwargs['name']))
        else:
            g.export(os.path.join(outdir,''.join(self.root)+'_tri.pdf'))
        return g
    
    @property
    def results(self):
        re=[]
        for k in range(self._n):
            n=len(self.Samp[k].getParamNames().names)
            pnames=self.Samp[k].getParamNames().names
            plt.figure(figsize=(10,6+(n-1)), dpi=90)
            plt.axes([0.025,0.025,0.95,0.95])
            plt.xticks([]), plt.yticks([])
            plt.text(0.1,0.9,'The results of "{0}" are:'.format(self.root[k].replace('_',' ')), fontsize=18)
            plt.text(0.2,0.83,'$1\sigma$',fontsize=14)
            plt.text(0.7,0.83,'$2\sigma$',fontsize=14)
            size = 20
            for i in range(n):
#                eqs="${{{0}}}^{{+{1}}}_{{-{2}}}$".format(round(theta_fact[i][0],5),round(theta_fact[i][1],5),round(theta_fact[i][2],5))
#                tt="${0}$ = {1}".format(pnames[i].label,eqs)
                tt='$%s$'%(self.Samp[k].getInlineLatex(pnames[i].name,limit=1))
                re.append(tt)
                tt2='$%s$'%(self.Samp[k].getInlineLatex(pnames[i].name,limit=2))
                x,y = (0.1,0.75-i*0.12)
                xx,yy=(0.6,0.75-i*0.12)
                plt.text(x,y,tt, fontsize=size)
                plt.text(xx,yy,tt2, fontsize=size)
            if self.aic_g:
                aic="$\mathrm{{AIC}}$=${0}$".format(round(self.minkaf[k]+2.0*n,3))
                bic="$\mathrm{{BIC}}$=${0}$".format(round(self.minkaf[k]+n*np.log(self.data_num[k]),3))
                kafm="$\chi^2_{{min}}$=${0}$".format(round(self.minkaf[k],3))
                dof="$\chi^2_{{min}}/d.o.f.$=${0}$".format(round(self.minkaf[k]/(n+self.data_num[k]),3))
                plt.text(0.1,0.8-(n+1)*0.11,kafm,fontsize=size)
                plt.text(0.6,0.8-(n+1)*0.11,dof,fontsize=size)
                plt.text(0.1,0.8-(n+2)*0.11,aic,fontsize=size)
                plt.text(0.6,0.8-(n+2)*0.11,bic,fontsize=size)
            plt.savefig(outdir+self.root[k]+'_results.png',dpi=300)
        for i in range(self._n):
            print(',  '.join(re[i:i+n])+'\n')
        return re

    @property
    def results2(self):
        re=[]
        for k in range(self._n):
            n=len(self.Samp[k].getParamNames().names)
            pnames=self.label_name
            plt.figure(figsize=(10,6+(n-1)), dpi=90)
            plt.axes([0.025,0.025,0.95,0.95])
            plt.xticks([]), plt.yticks([])
            plt.text(0.1,0.9,'The results of "{0}" are:'.format(self.root[k].replace('_',' ')), fontsize=18)
            plt.text(0.5,0.83,'$1\sigma$',fontsize=14)
            # plt.text(0.7,0.83,'$2\sigma$',fontsize=14)
            size = 20
            for i in range(n):
                mcmc = np.percentile(self.Samp[k].samples[:, i], [16, 50, 84])
                q = np.diff(mcmc)
                txt = "$\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$"
                tt = txt.format(mcmc[1], q[0], q[1], pnames[i])
                # tt='$%s$'%(self.Samp[k].getInlineLatex(pnames[i].name,limit=1))
                re.append(tt)
                x,y = (0.2,0.75-i*0.12)
                plt.text(x,y,tt, fontsize=size)
            if self.aic_g:
                aic="$\mathrm{{AIC}}$=${0}$".format(round(self.minkaf[k]+2.0*n,3))
                bic="$\mathrm{{BIC}}$=${0}$".format(round(self.minkaf[k]+n*np.log(self.data_num[k]),3))
                kafm="$\chi^2_{{min}}$=${0}$".format(round(self.minkaf[k],3))
                dof="$\chi^2_{{min}}/d.o.f.$=${0}$".format(round(self.minkaf[k]/(n+self.data_num[k]),3))
                plt.text(0.1,0.8-(n+1)*0.11,kafm,fontsize=size)
                plt.text(0.6,0.8-(n+1)*0.11,dof,fontsize=size)
                plt.text(0.1,0.8-(n+2)*0.11,aic,fontsize=size)
                plt.text(0.6,0.8-(n+2)*0.11,bic,fontsize=size)
            plt.savefig(outdir+self.root[k]+'_results2.png',dpi=300)
        for i in range(self._n):
            print(',  '.join(re[i:i+n])+'\n')
        return re

class Fisherplot(MCplot):
    def __init__(self,mean,Cov,labels,lengend='',nsample=1000000):
        self.mean = mean
        self.Cov = Cov
        self.param_names = labels
        self.lengend = [lengend]
        self.nsample = nsample
        self._n = 1
        self.root = [lengend]
        self.init()
        self.aic_g=False
    
    def init(self):
        # self.param_names=[x.replace('H_0','H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]') for x in self.param_names]
        gauss=GaussianND(self.mean, self.Cov ,names = self.param_names, labels =self.param_names)
        self.Samp = [gauss.MCSamples(self.nsample)]
    
    def addCov(self,mean,Cov,lengend):
        gauss=GaussianND(mean, Cov ,names = self.param_names, labels =self.param_names)
        self.Samp.append(gauss.MCSamples(self.nsample))
        self.lengend.append(lengend)
        self._n = len(self.Samp)
        self.root.append(lengend)
    
    def addChains(self,Chains,ignore_rows=0.3):
        root=list(np.asarray(Chains)[:,0])
        lengend=list(np.asarray(Chains)[:,1])
        n = len(Chains)
        for i in range(n):
            savefile_name='./chains/'+root[i]+'.npy'
            samples,theta_name,theta_fit,theta_fact,minkaf,data_num,ranges=np.load(savefile_name, allow_pickle=True)
            self.Samp.append(MCSamples(samples=samples,names = theta_name, labels = theta_name,ranges=ranges,settings={'ignore_rows':ignore_rows}))
        self.param_names=[]
        for na in self.Samp[-1].getParamNames().names:
            self.param_names.append(na.name)
        self.lengend+=lengend
        self._n = len(self.Samp)
        self.root+=root
    


class CMCplot(MCplot):
    def __init__(self,Chains,ignore_rows=0.3):
        self.root=list(np.asarray(Chains)[:,0])
        self.lengend=list(np.asarray(Chains)[:,1])
        self.aic_g=False
        self.Samp=[]
        self._n = len(Chains)
        self.minkaf=np.zeros(self._n)
        self.data_num=np.zeros(self._n)
        for i in list(range(self._n)):
            self.Samp.append(loadMCSamples('./chains/'+self.root[i], settings={'ignore_rows':ignore_rows}))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
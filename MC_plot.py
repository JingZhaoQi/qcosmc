# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 15:52:29 2018

@author: qijingzhao
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from getdist import MCSamples
from getdist import loadMCSamples
from getdist import plots
from .FigStyle import qstyle
qstyle(1)
colors=['#348ABD', '#A60628', '#467821', '#7A68A6',  '#E24A33' ,'#ffb3a6', '#188487']
lss=['-','--','-.',':']
outdir='./results/'

class MCplot(object):
    def __init__(self,Chains):
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
            samples,self.theta_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges=np.load(savefile_name)
            self.Samp.append(MCSamples(samples=samples,names = self.theta_name, labels = self.theta_name,ranges=ranges))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
    
    def rename(self,new_name):
        for i in range(self._n):
            savefile_name='./chains/'+self.root[i]+'.npy'
            samples,self.theta_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges=np.load(savefile_name)
            np.save(savefile_name,(samples,new_name,self.theta_fit,self.theta_fact,self.minkaf[i],self.data_num[i],ranges))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
    
    def plot1D(self,n,colorn=0,width_inch=8,**kwargs):
        g = plots.getSinglePlotter(width_inch=width_inch)
        g.plot_1d(self.Samp,self.param_names[n-1],ls=lss,colors=colors[colorn:colorn+self._n],lws=[1.5]*self._n,**kwargs)
        ax=plt.gca()
        if all(self.lengend):
            leg = ax.legend(self.lengend,loc=1,fontsize=16)
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'],lw=1.5)
#        if 'x_bands' in kwargs:
#            g.add_x_bands(0,0.01)
        if 'xaxis' in kwargs:
            ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['xaxis']))
        plt.tight_layout()
        g.export(os.path.join(outdir+''.join(self.root)+self.param_names[n-1].replace('\\','')+'_1D.pdf'))
    
    
    def plot2D(self,pp,colorn=0,contour_num=2,width_inch=8,**kwargs):
        g = plots.getSinglePlotter(width_inch=width_inch,**kwargs)
        g.settings.num_plot_contours = contour_num
        g.settings.axes_fontsize = 14
        g.settings.lab_fontsize = 20
        g.plot_2d(self.Samp,self.param_names[pp[0]-1],self.param_names[pp[1]-1],filled=True,colors=colors[colorn:colorn+self._n],**kwargs)
        if all(self.lengend):
            g.add_legend(self.lengend,colored_text=True, fontsize=18)
#        if 'lims' in kwargs:
#            [xmin, xmax, ymin, ymax]=kwargs['lims']
#            plt.xlim(xmin, xmax)
#            plt.ylim(ymin, ymax)
        if 'x_marker' in kwargs:
            g.add_x_marker(kwargs['x_marker'],lw=1.5)
        plt.tight_layout()
        g.export(os.path.join(outdir,''.join(self.root)+'_2D.pdf'))


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
        g = plots.getSubplotPlotter(width_inch=9)
        g.settings.num_plot_contours = contour_num
        g.settings.legend_fontsize = 18
        g.settings.axes_fontsize = 12
        g.settings.lab_fontsize = 16
        g.settings.figure_legend_frame = False
        g.settings.alpha_filled_add=0.8
        g.triangle_plot(self.Samp,t_name,filled_compare=True,legend_labels=self.lengend,contour_colors=colorss,**kwargs)
        if 'tline' in kwargs:
            for ax in g.subplots[:,0]:
                ax.axvline(kwargs['tline'], color='red', ls='--',alpha=0.5)
#        plt.tight_layout()
        if 'xaxis1' in kwargs:
            for ax in g.subplots[:,0]:
                ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['xaxis1']))
        if 'xaxis2' in kwargs:
            for ax in g.subplots[:,1]:
                ax.xaxis.set_major_locator(plt.MultipleLocator(kwargs['xaxis2']))
        g.export(os.path.join(outdir,''.join(self.root)+'_tri.pdf'))
    
    @property
    def results(self):
        for k in range(self._n):
            n=len(self.Samp[k].getParamNames().names)
            pnames=self.Samp[k].getParamNames().names
            re=[]
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
        return re


class CMCplot(MCplot):
    def __init__(self,Chains,ignore_rows=0.3):
        self.root=list(np.asarray(Chains)[:,0])
        self.lengend=list(np.asarray(Chains)[:,1])
        self.aic_g=False
        self.Samp=[]
        self._n = len(Chains)
        self.minkaf=np.zeros(self._n)
        self.data_num=np.zeros(self._n)
        for i in range(self._n):
            self.Samp.append(loadMCSamples('./chains/'+self.root[i], settings={'ignore_rows':ignore_rows}))
        self.param_names=[]
        for na in self.Samp[0].getParamNames().names:
            self.param_names.append(na.name)
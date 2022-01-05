# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 15:16:51 2018

@author: Qijingzhao
"""
#import numpy as np
import seaborn as sns
import matplotlib as mpl
from getdist import MCSamples
from getdist import plots
import numpy as np
size={ 'figure.figsize'   : (8, 6),
    'figure.dpi'       : 100,
    'xtick.labelsize' : 11,
    'ytick.labelsize' : 11,
    'axes.labelsize':18,
    'legend.fontsize'  : 18,
    'axes.titlesize' : 18}
aaa={'axes.axisbelow': True,
    'axes.edgecolor': '.0',
    'axes.titlesize' : 18,
    'axes.facecolor': '1',
    'axes.grid': False,
    'axes.labelcolor': '.0',
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'figure.figsize'   : (8, 6),
    'figure.dpi'       : 100,
    'lines.linewidth'   : 1,
    'font.family': [u'sans-serif'],
    'font.sans-serif': [u'Arial',
    u'DejaVu Sans',
    u'Liberation Sans',
    u'Bitstream Vera Sans',
    u'sans-serif'],
    'grid.color': '.8',
    'grid.linestyle': u'-',
    'image.cmap': u'rocket',
    'legend.scatterpoints': 1,
    'lines.solid_capstyle': u'round',
    'text.color': '0',
    #'axes.labelsize': 20,
    'axes.labelsize': 'large',
    'xtick.color': '0.',
    #'xtick.top'     : True,
    'xtick.direction': u'in',
    'xtick.major.size': 4 ,
    'xtick.minor.size': 2,
    'xtick.labelsize' : 13,
    'ytick.color': '.0',
    #'ytick.right'     : True,
    'ytick.direction': u'in',
    'ytick.major.size': 4 ,
    'ytick.minor.size': 2,
    'ytick.labelsize' : 13,
    'legend.fancybox'      : False,
    'legend.numpoints'    : 1,
    'legend.fontsize'  : 16,
    'legend.frameon' : False}

def qplt(*args, **kwargs):
    s=''
    if 'xscale' in kwargs:
        mpl.pyplot.xscale(kwargs['xscale'])
        del kwargs['xscale']
    if 'yscale' in kwargs:
        mpl.pyplot.yscale(kwargs['yscale'])
        del kwargs['yscale']
    if 'xlabel' in kwargs:
        mpl.pyplot.xlabel('%s'%kwargs['xlabel'])
        del kwargs['xlabel']
    if 'ylabel' in kwargs:
        mpl.pyplot.ylabel('%s'%kwargs['ylabel'])
        del kwargs['ylabel']
    if 'lims' in kwargs:
        [xmin, xmax, ymin, ymax]=kwargs['lims']
        mpl.pyplot.xlim(xmin, xmax)
        mpl.pyplot.ylim(ymin, ymax)
        del kwargs['lims']
    if 'title' in kwargs:
        mpl.pyplot.title(kwargs['title'])
        del kwargs['title']
    if 'save' in kwargs:
        s=kwargs['save']
        del kwargs['save']
    if 'xmin' in kwargs:
        xmin=mpl.pyplot.MultipleLocator(kwargs['xmin'])
        del kwargs['xmin']
    else: xmin=mpl.ticker.AutoMinorLocator()
    if 'ymin' in kwargs:
        ymin=mpl.pyplot.MultipleLocator(kwargs['ymin'])
        del kwargs['ymin']
    else: ymin=mpl.ticker.AutoMinorLocator()
    mpl.pyplot.plot(*args, **kwargs)
    ax=mpl.pyplot.gca()
#    ax.xaxis.set_minor_locator(xmin)
#    ax.yaxis.set_minor_locator(ymin)
    if 'label' in kwargs:
        ax=mpl.pyplot.gca()
        leg = ax.legend(loc=0,numpoints=1)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
    mpl.pyplot.tight_layout()
    if s:
        mpl.pyplot.savefig(s)

def qstyle(tex=False,rc={}):
#    sns.reset_defaults()
#    sns.set(style="ticks")
    color=['#348ABD', '#7A68A6',  '#E24A33', '#467821' ,'#ffb3a6', '#188487', '#A60628']
    # sns.color_palette(color)
    mpl.rc('text', usetex=tex)
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
    aaa.update(rc)
    sns.set_style(aaa)
    mpl.rcParams['xtick.labelsize'] = 11
    mpl.rcParams['ytick.labelsize'] = 11
    mpl.rcParams['axes.labelsize'] = 16
    return color

def snstyle(tex=False,stl='ticks',rc={}):
    '''
    darkgrid, whitegrid, dark, white, ticks
    '''
#    sns.reset_defaults()
    mpl.rc('text', usetex=tex)
    sns.set_style(stl)
    size.update(rc)
    sns.set_style(size)

def plot2D(rdic,par=[1,2],tex=1):
    snstyle(tex)
    rdic=np.asarray(rdic)
    root=list(rdic[:,0])
    lengend=list(rdic[:,1])
    rn=len(root)
    Samp=[]
    minkaf=np.zeros(rn)
    data_num=np.zeros(rn)
    for i in range(rn):
        savefile_name='./chains/'+root[i]+'.npy'
        samples,theta_name,theta_fit,theta_fact,minkaf[i],data_num[i]=np.load(savefile_name)
        Samp.append(MCSamples(samples=samples,names = theta_name, labels = theta_name))    
    pnames=Samp[0].getParamNames().names
    rn=len(root)
    g = plots.getSinglePlotter(width_inch=7)
    #samples.updateSettings({'contours': [0.68, 0.95, 0.99]})
    #g.settings.num_plot_contours = 3
    g.settings.lab_fontsize=18
    g.settings.axes_fontsize = 14
    g.plot_2d(Samp,pnames[par[0]-1].name,pnames[par[1]-1].name)
    for i in range(rn):
        sns.kdeplot(Samp[i].samples[:,par[0]-1],Samp[i].samples[:,par[1]-1],cmap="Blues", shade=True, shade_lowest=False)
    g.add_legend(lengend,colored_text=True, fontsize=18)
    mpl.pyplot.tight_layout()


  







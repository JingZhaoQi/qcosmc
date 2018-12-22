# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:15:38 2017

@author: jingzhao
"""
from qicosmc.cos_models import wCDM
from qicosmc.MCMC import MCMC_cos
''' 
可使用数据如下，如要使用，在MCMC_cos()里面填入
Ues_***=1即可。
Ues_JLA,Ues_BAO,Ues_OHD,Ues_CMBshift,
Ues_CMB,Ues_SN,Ues_QSO,Ues_fs8,Ues_SL

params里面依次为["参数名字"，“初始值”，“最小值”，“最大值”]
如需增加参数，复制一行，直接按照格式填入即可。
'''

params=(['\Omega_m',0.3,0,1],
        ['w',-1,-2,0],
        )


def model(theta):
    om,w=theta
    return wCDM(om,w)

Chains_name='wcdm_OHD'
MCMC_cos(params,model,Chains_name,Ues_OHD=1)
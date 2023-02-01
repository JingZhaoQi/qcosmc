# Supernovae likelihood, from CosmoMC's JLA module. For Pantheon and JLA Supernovae,
#  History:
#  Written by Alex Conley, Dec 2006
#   aconley, Jan 2007: The OpenMP stuff was causing massive slowdowns on
#      some processors (ones with hyperthreading), so it was removed
#   aconley, Jul 2009: Added absolute distance support
#   aconley, May 2010: Added twoscriptm support
#   aconley, Apr 2011: Fix some non standard F90 usage.  Thanks to
#                       Zhiqi Huang for catching this.
#   aconley, April 2011: zhel, zcmb read in wrong order.  Thanks to
#                       Xiao Dong-Li and Shuang Wang for catching this
#   mbetoule, Dec 2013: adaptation to the JLA sample
#   AL, Mar 2014: updates for latest CosmoMC structure
#   AL, June 2014: updated JLA_marginalize=T handling so it should work
#   AL, March 2018: this python version

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
from getdist import IniFile
import io
import os

_twopi = 2 * np.pi

_marge_params = {'marge_steps': 7, 'step_width_alpha': 0.003, 'step_width_beta': 0.04,
                 'alpha_centre': 0.14, 'beta_centre': 3.123}
dataDir=os.path.dirname(os.path.abspath(__file__))+'/data/full_long.dataset'

class SN_likelihood(object):

    def __init__(self, dataset=dataDir, dataset_params={}, alpha_beta_names=['alpha', 'beta'],
                 marginalize=False, marginalize_params=_marge_params, precompute_covmats=True, silent=False):
        """

        :param dataset: .dataset file with settings
        :param dataset_params:  dictionary of any parameter to override in teh .dataset file
        :param alpha_beta_names: names of alpha and beta parameters if used and varied
        :param marginalize: Marginalize over alpha, beta by dumb grid integration (slow, but useful for importance sampling)
        :param marginalize_params: Dictionary of options for the grid marguinalization
        :param precompute_covmats: if marginalizing, pre-compute covariance inverses at expense of memory (~600MB).
        :param silent:  Don't print out stuff
        """

        def relative_path(tag):
            name = ini.string(tag).replace('data/', '').replace('Pantheon/', '')
            if ini.original_filename is not None:
                return os.path.join(os.path.dirname(ini.original_filename), name)
            return name

        # has_absdist = F, intrinsicdisp=0, idispdataset=False
        if not silent: print('loading: %s' % dataset)
        ini = IniFile(dataset)
        ini.params.update(dataset_params)
        self.name = ini.string('name')
        data_file = relative_path('data_file')
        self.twoscriptmfit = ini.bool('twoscriptmfit')
        if self.twoscriptmfit:
            scriptmcut = ini.float('scriptmcut', 10.)

        assert not ini.float('intrinsicdisp', 0) and not ini.float('intrinsicdisp0', 0)
        self.alpha_beta_names = alpha_beta_names
        if alpha_beta_names is not None:
            self.alpha_name = alpha_beta_names[0]
            self.beta_name = alpha_beta_names[1]

        self.marginalize = marginalize

        self.pecz = ini.float('pecz', 0.001)

        cols = None
        self.has_third_var = False

        if not silent:
            print('Supernovae name: %s' % self.name)
            print('Reading %s' % data_file)
        supernovae = {}
        self.names = []
        ix = 0
        with io.open(data_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '#' in line:
                    cols = line[1:].split()
                    for rename, new in zip(['mb', 'color', 'x1', '3rdvar', 'd3rdvar', 'cov_m_s', 'cov_m_c', 'cov_s_c'],
                                           ['mag', 'colour', 'stretch', 'third_var', 'dthird_var', 'cov_mag_stretch',
                                            'cov_mag_colour', 'cov_stretch_colour']):
                        if rename in cols:
                            cols[cols.index(rename)] = new
                    self.has_third_var = 'third_var' in cols
                    zeros = np.zeros(len(lines) - 1)
                    self.third_var = zeros.copy()
                    self.dthird_var = zeros.copy()
                    self.set = zeros.copy()
                    for col in cols:
                        setattr(self, col, zeros.copy())
                elif line.strip():
                    if cols is None: raise Exception('Data file must have comment header')
                    vals = line.split()
                    for i, (col, val) in enumerate(zip(cols, vals)):
                        if col == 'name':
                            supernovae[val] = ix
                            self.names.append(val)
                        else:
                            getattr(self, col)[ix] = np.float64(val)
                    ix += 1

        self.z_var = self.dz ** 2
        self.mag_var = self.dmb ** 2
        self.stretch_var = self.dx1 ** 2
        self.colour_var = self.dcolor ** 2
        self.thirdvar_var = self.dthird_var ** 2
        self.nsn = ix
        if not silent: print('Number of SN read: %s ' % self.nsn)

        if self.twoscriptmfit and not self.has_third_var:
            raise Exception('twoscriptmfit was set but thirdvar information not present')

        if ini.bool('absdist_file'): raise Exception('absdist_file not supported')

        covmats = ['mag', 'stretch', 'colour', 'mag_stretch', 'mag_colour', 'stretch_colour']
        self.covs = {}
        for name in covmats:
            if ini.bool('has_%s_covmat' % name):
                if not silent: print('Reading covmat for: %s ' % name)
                self.covs[name] = self._read_covmat(relative_path('%s_covmat_file' % name))

        self.alphabeta_covmat = len(self.covs.items()) > 1 or self.covs.get('mag', None) is None
        self._last_alpha = np.inf
        self._last_beta = np.inf
        if alpha_beta_names is None and not marginalize: raise ValueError('Must give alpha, beta')
        assert self.covs

        # jla_prep
        zfacsq = 25.0 / np.log(10.0) ** 2
        self.pre_vars = self.mag_var + zfacsq * self.pecz ** 2 * (
                (1.0 + self.zcmb) / (self.zcmb * (1 + 0.5 * self.zcmb))) ** 2

        if self.twoscriptmfit:
            A1 = np.zeros(self.nsn)
            A2 = np.zeros(self.nsn)
            A1[self.third_var <= scriptmcut] = 1
            A2[self.third_var > scriptmcut] = 1
            has_A1 = np.any(A1)
            has_A2 = np.any(A2)
            if not has_A1:
                # swap
                A1 = A2
                A2 = np.zeros(self.nsn)
                has_A2 = False

            if not has_A2:
                self.twoscriptmfit = False
            self.A1 = A1
            self.A2 = A2

        if marginalize:
            self.marge_params = _marge_params.copy()
            self.marge_params.update(marginalize_params)
            self.step_width_alpha = self.marge_params['step_width_alpha']
            self.step_width_beta = self.marge_params['step_width_beta']
            _marge_steps = self.marge_params['marge_steps']
            self.alpha_grid = np.empty((2 * _marge_steps + 1) ** 2)
            self.beta_grid = self.alpha_grid.copy()
            _int_points = 0
            for alpha_i in range(-_marge_steps, _marge_steps + 1):
                for beta_i in range(-_marge_steps, _marge_steps + 1):
                    if alpha_i ** 2 + beta_i ** 2 <= _marge_steps ** 2:
                        self.alpha_grid[_int_points] = self.marge_params[
                                                           'alpha_centre'] + alpha_i * self.step_width_alpha
                        self.beta_grid[_int_points] = self.marge_params['beta_centre'] + beta_i * self.step_width_beta
                        _int_points += 1
            if not silent: print('Marignalizing alpha, beta over %s points' % _int_points)
            self.marge_grid = np.empty(_int_points)
            self.int_points = _int_points
            self.alpha_grid = self.alpha_grid[:_int_points]
            self.beta_grid = self.beta_grid[:_int_points]
            self.invcovs = np.empty(_int_points, dtype=np.object)
            if precompute_covmats:
                for i, (alpha, beta) in enumerate(zip(self.alpha_grid, self.beta_grid)):
                    self.invcovs[i] = self.inverse_covariance_matrix(alpha, beta)

        elif not self.alphabeta_covmat:
            self.inverse_covariance_matrix()

    def _read_covmat(self, filename):
        cov = np.loadtxt(filename)
        if np.isscalar(cov[0]) and cov[0] ** 2 + 1 == len(cov):
            cov = cov[1:]
        return cov.reshape((self.nsn, self.nsn))

    def inverse_covariance_matrix(self, alpha=0, beta=0):
        if 'mag' in self.covs:
            invcovmat = self.covs['mag'].copy()
        else:
            invcovmat = 0
        if self.alphabeta_covmat:
            if np.isclose(alpha, self._last_alpha) and np.isclose(beta, self._last_beta):
                return self.invcov
            self._last_alpha = alpha
            self._last_beta = beta

            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            if 'stretch' in self.covs:
                invcovmat += alphasq * self.covs['stretch']
            if 'colour' in self.covs:
                invcovmat += betasq * self.covs['colour']
            if 'mag_stretch' in self.covs:
                invcovmat += 2 * alpha * self.covs['mag_stretch']
            if 'mag_colour' in self.covs:
                invcovmat -= 2 * beta * self.covs['mag_colour']
            if 'stretch_colour' in self.covs:
                invcovmat -= 2 * alphabeta * self.covs['stretch_colour']

            delta = self.pre_vars + alphasq * self.stretch_var + \
                    + betasq * self.colour_var + 2.0 * alpha * self.cov_mag_stretch \
                    - 2.0 * beta * self.cov_mag_colour \
                    - 2.0 * alphabeta * self.cov_stretch_colour
        else:
            delta = self.pre_vars
        np.fill_diagonal(invcovmat, invcovmat.diagonal() + delta)
        self.invcov = np.linalg.inv(invcovmat)
        return self.invcov

    def alpha_beta_like(self, lumdists, alpha=0, beta=0, invcovmat=None):
        if self.alphabeta_covmat:
            alphasq = alpha * alpha
            betasq = beta * beta
            alphabeta = alpha * beta
            invvars = 1.0 / (self.pre_vars + alphasq * self.stretch_var
                             + betasq * self.colour_var
                             + 2.0 * alpha * self.cov_mag_stretch
                             - 2.0 * beta * self.cov_mag_colour
                             - 2.0 * alphabeta * self.cov_stretch_colour)
            wtval = np.sum(invvars)
            estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = self.mag - lumdists + alpha * self.stretch \
                      - beta * self.colour - estimated_scriptm
            if invcovmat is None:
                invcovmat = self.inverse_covariance_matrix(alpha, beta)
        else:
            invvars = 1.0 / self.pre_vars
            wtval = np.sum(invvars)
            estimated_scriptm = np.sum((self.mag - lumdists) * invvars) / wtval
            diffmag = self.mag - lumdists - estimated_scriptm
            invcovmat = self.invcov

        invvars = invcovmat.dot(diffmag)
        amarg_A = invvars.dot(diffmag)
        if self.twoscriptmfit:
            # could simplify this..
            amarg_B = invvars.dot(self.A1)
            amarg_C = invvars.dot(self.A2)
            invvars = invcovmat.dot(self.A1)
            amarg_D = invvars.dot(self.A2)
            amarg_E = invvars.dot(self.A1)
            invvars = invcovmat.dot(self.A2)
            amarg_F = invvars.dot(self.A2)

            tempG = amarg_F - amarg_D * amarg_D / amarg_E
            assert tempG >= 0
            chi2 = amarg_A + np.log(amarg_E / _twopi) + \
                   np.log(tempG / _twopi) - amarg_C * amarg_C / tempG - \
                   amarg_B * amarg_B * amarg_F / (amarg_E * tempG) + 2.0 * amarg_B * amarg_C * amarg_D / (
                           amarg_E * tempG)

        else:
            amarg_B = np.sum(invvars)
            amarg_E = np.sum(invcovmat)
            chi2 = amarg_A + np.log(amarg_E / _twopi) - amarg_B ** 2 / amarg_E
        return chi2 / 2

    def get_redshifts(self):
        return self.zcmb

    def loglike(self, angular_diameter_distances, data_params={}):
        assert len(angular_diameter_distances) == len(self.zcmb)

        lumdists = 5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * angular_diameter_distances)
        if self.marginalize:
            # Should parallelize this loop
            for i in range(self.int_points):
                self.marge_grid[i] = self.alpha_beta_like(lumdists, self.alpha_grid[i], self.beta_grid[i],
                                                          invcovmat=self.invcovs[i])
            grid_best = np.min(self.marge_grid)
            return grid_best - np.log(np.sum(np.exp(-self.marge_grid[self.marge_grid != np.inf]
                                                    + grid_best)) * self.step_width_alpha * self.step_width_beta)
        else:
            if self.alphabeta_covmat:
                return self.alpha_beta_like(lumdists, data_params[self.alpha_name],
                                            data_params[self.beta_name])
            else:
                return self.alpha_beta_like(lumdists)




default_data_file = os.path.dirname(os.path.abspath(__file__))+"/data/Pantheon+SH0ES.dat"
default_covmat_file = os.path.dirname(os.path.abspath(__file__))+"/data/Pantheon+SH0ES_STAT+SYS.cov"


class PantheonPlus(object):
    def __init__(self,cut_redshift=0.01,data_type='pantheon'):
        self.cut_redshift=cut_redshift
        # self.max_redshift=max_redshift
        self.datatype=data_type
        self.read_data()
        self.build_data()
        self.build_cov()
        
        
    def read_data(self):
        self.data = pd.read_csv(default_data_file,delim_whitespace=True)
    
    def build_data(self):
        data = self.data
        self.origlen = len(data)
        
        # self.ww = ((self.cut_redshift<data['zHD'])&(data['zHD']<=self.max_redshift))
        if self.datatype=='SH0ES':
            self.ww = (data['zHD']>self.cut_redshift) | (np.array(data['IS_CALIBRATOR'],dtype=bool))
            self.shdata=np.array(data['IS_CALIBRATOR'][self.ww],dtype=bool)
            self.is_calibrator = data['IS_CALIBRATOR'][self.ww]
            self.cepheid_distance = data['CEPH_DIST'][self.ww]
            print('Data type is Pantheon+SH0ES')
        else:
            self.ww = (data['zHD']>self.cut_redshift)
            self.cepheid_distance = data['CEPH_DIST'][self.ww]
            self.shdata=np.zeros(len(self.cepheid_distance),dtype=bool)
            print('Data type is Pantheon')

        self.zcmb = data['zHD'][self.ww] #use the vpec corrected redshift for zCMB 
        self.zhel = data['zHEL'][self.ww]
        self.m_obs = data['m_b_corr'][self.ww]
        print('The cut redshift is z=%s, and the number of data is %s.'%(self.cut_redshift,len(self.zcmb)))
        return self.zcmb, self.m_obs
    
    # @timer
    def build_covariance(self):
        """Run once at the start to build the covariance matrix for the data"""
        # print("Loading covariance from {}".format(filename))

        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This function reads in the file and the nasty for loops trim down the covariance
        # to match the only rows of data that are used for cosmology

        f = open(default_covmat_file)
        n = int(len(self.zCMB))
        C = np.zeros((n,n))
        ii = -1
        jj = -1
        for i in range(self.origlen):
            jj = -1
            if self.ww[i]:
                ii += 1
            for j in range(self.origlen):
                if self.ww[j]:
                    jj += 1
                val = float(f.readline())
                if self.ww[i]:
                    if self.ww[j]:
                        C[ii,jj] = val
        f.close()

        print('Done')
        return C
    
    # @timer
    def build_cov(self):
        """Run once at the start to build the covariance matrix for the data"""
        # print("Loading covariance from {}".format(filename))

        # The file format for the covariance has the first line as an integer
        # indicating the number of covariance elements, and the the subsequent
        # lines being the elements.
        # This function reads in the file and the nasty for loops trim down the covariance
        # to match the only rows of data that are used for cosmology
        ff=np.loadtxt(default_covmat_file)
        # f1=ff[:-1].reshape(self.origlen,self.origlen)
        f1=ff[1:].reshape(self.origlen,self.origlen)
        self.cov=f1[self.ww,:][:,self.ww]
        self.incov=np.linalg.inv(self.cov)
        # return C
    
    def get_redshifts(self):
        return self.zcmb
    
    def chi2(self,model,Mb):
        mu_th = 5 * np.log10((1 + self.zhel) * (1 + self.zcmb) * model.MC_DA(self.zcmb))+25
        mu_th[self.shdata]=self.cepheid_distance[self.shdata]
        delta_mu=mu_th+Mb-self.m_obs
        return np.dot(delta_mu,np.dot(self.incov,delta_mu))
    
    def chi2_comoving(self,dc,Mb):
        mu_th=5 * np.log10((1 + self.zhel) * dc(self.zcmb))+25
        mu_th[self.shdata]=self.cepheid_distance[self.shdata]
        delta_mu=mu_th+Mb-self.m_obs
        return np.dot(delta_mu,np.dot(self.incov,delta_mu))
                           


if __name__ == "__main__":
    import time


    def fit(z):
        return -338.65487197 * z ** 4 + 1972.59141641 * z ** 3 - 4310.60442428 * z ** 2 + 4357.72542145 * z


    # Pantheon (alpha and beta not used - no nuisance parameters), fast
    like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\Pantheon\full_long.dataset')
    zs = like.get_redshifts()
    start = time.time()
    chi2 = like.loglike(fit(zs)) * 2
    print('Pantheon chi^2: %.2f, expected 1054.56' % chi2)
    print('Likelihood execution time:', time.time() - start)
    assert np.isclose(chi2, 1054.557083)
    print('')

    # JLA with alpha, beta parameters passed in, fairly fast (one matrix inversion)
    like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', marginalize=False)
    zs = like.get_redshifts()
    start = time.time()
    chi2 = like.loglike(fit(zs), {'alpha': 0.1325237, 'beta': 2.959805}) * 2
    print('Likelihood execution time:', time.time() - start)
    print('JLA chi^2: %.2f, expected 716.23' % chi2)
    assert np.isclose(chi2, 716.2296141)
    print('')

    # JLA marginalized over alpha, beta, e.g. for use in importance sampling with no nuisance parameters.
    # Quite fast as inverses precomputed. Note normalization is not same as for alpha, beta varying.
    like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', marginalize=True)
    zs = like.get_redshifts()
    start = time.time()
    chi2 = like.loglike(fit(zs)) * 2
    print('Likelihood execution time:', time.time() - start)
    print('JLA marged chi^2: %.2f, expected 720.00' % chi2)
    assert np.isclose(chi2, 720.0035394)

    # as above, but very slow (but lower memory) using non-precomputed inverses (and non-threaded in python)
    like = SN_likelihood(r'C:\Work\Dist\git\cosmomcplanck\data\jla.dataset', precompute_covmats=False, marginalize=True)
    zs = like.get_redshifts()
    start = time.time()
    chi2 = like.loglike(fit(zs)) * 2
    print('Likelihood execution time:', time.time() - start)
    print('JLA marged chi^2: %.2f, expected 720.00' % chi2)
    assert np.isclose(chi2, 720.0035394)

"""
We start with a bunch of standard imports..
"""


import numpy as np
import pandas as pd
import math
import emcee
import sys
import pickle
from sklearn.neighbors import KernelDensity
import os

"""
Next, we import the KDELikelihood implemented in `lenstronomy` in order to properly sample lenstronomy's 2D likelihood (Dd versus Ddt, more on that below).

We also import cosmological models from `astropy`. 
These will be used to compute cosmological angular diameter distances from cosmological parameters in various cosmologies.
"""


from lenstronomy.Cosmo.kde_likelihood import KDELikelihood

"""
The output of the lens modeling are angular diameter distances; depending on the modeling code used, 
it is either the time-delay distance Ddt alone (using `GLEE`) or a joint time-delay distance Ddt versus observer-deflector angular diameter distance Dd (using `lenstronomy`).

In order to easily merge the output of these two softwares - and possibly combine with other softwares in the future - 
we create the classes GLEELens and LenstronomyLens that we initiate with the angular diameter distances posteriors predicted by 
the modeling codes and whose main goal is to evalute the likelihood of a chosen cosmological model with respect to the input posteriors of the lens.

    GLEE's time-delay distance posteriors are usually well fitted by a skewed log-normal likelihood. 
    The analytical fit parameters are thus the input parameters of a GLEELens object. 
    Alternatively, one can fit the angular diameter distance with a KDELikelihood if the skewed log-normal fit does not properly capture the distribution.

    Lenstronomy's time-delay distance and observer-deflector angular diameter distance joint posteriors are fitted with a KDELikelihood. 
    They are the input parameters of a LenstronomyLens object.
"""

class StrongLensSystem(object):
    """
    This is a parent class, common to all lens modeling code outputs.
    It stores the "physical" parameters of the lens (name, redshifts, ...)
    """
    def __init__(self, name, zlens, zsource, longname=None):
        self.name = name
        self.zlens = zlens
        self.zsource = zsource
        self.longname = longname

    def __str__(self):
        return "%s\n\tzl = %f\n\tzs = %f" % (self.name, self.zlens, self.zsource)


class GLEELens(StrongLensSystem):
    """
    This class takes the output of GLEE (Ddt distribution) from which it evaluates the likelihood of a Ddt (in Mpc) predicted in a given cosmology.

    The default likelihood follows a skewed log-normal distribution. You can also opt for a normal distribution for testing purposes. 
    In case no analytical form fits the Ddt distribution well, 
    one can use a KDE log_likelihood instead - either fitted on the whole Ddt distribution (slow) or on a binned version of it (faster, no significant loss of precision).
    
    You can now also give mu,sigma and lambda parameter of the skewed likelihood for Dd. 
    """
    def __init__(self, name, zlens, zsource,
                 loglikelihood_type="normal_analytical",
                 mu=None, sigma=None, lam=None, explim=100.,
                 ddt_samples=None, dd_samples=None, weights = None, kde_kernel=None,
                 bandwidth=20, nbins_hist=200,
                 longname=None, mu_Dd=None, sigma_Dd=None, lam_Dd=None):

        StrongLensSystem.__init__(self, name=name, zlens=zlens, zsource=zsource, longname=longname)
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.explim = explim
        self.loglikelihood_type = loglikelihood_type
        self.ddt_samples = ddt_samples
        self.dd_samples = dd_samples
        if weights is None :
            if self.ddt_samples is not None :
                self.weights = np.ones(len(self.ddt_samples))
            else :
                self.weights = None
        else :
            self.weights = weights
        self.kde_kernel = kde_kernel
        self.bandwidth = bandwidth
        self.nbins_hist = nbins_hist
        self.mu_Dd = mu_Dd
        self.sigma_Dd = sigma_Dd
        self.lam_Dd = lam_Dd

        # do it only once at initialisation
        if loglikelihood_type == "hist_lin_interp":
            self.vals, self.bins = np.histogram(self.ddt_samples["ddt"], bins=self.nbins_hist, weights=self.weights)

        self.init_loglikelihood()

    def sklogn_analytical_likelihood(self, ddt):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a skewed log-normal distribution.
        """
        # skewed lognormal distribution with boundaries
        if (ddt <= self.lam) or ((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2) > self.explim):
            return -np.inf
        else:
            llh = math.exp(-((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2))) / (math.sqrt(2 * math.pi) * (ddt - self.lam) * self.sigma)

            if np.isnan(llh):
                return -np.inf
            else:
                return np.log(llh)

    def sklogn_analytical_likelihood_Dd(self, ddt, dd):
        """
        Evaluates the likelihood of a time-delay distance ddt and angular diameter distance Dd(in Mpc) against the model predictions, using a skewed log-normal distribution for both ddt and dd. The two distributions are asssumed independant and can be combined
        """
        # skewed lognormal distribution with boundaries
        if (ddt < self.lam) or ((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2) > self.explim) or (dd < self.lam_Dd) or ((-self.mu_Dd + math.log(dd - self.lam_Dd)) ** 2 / (2. * self.sigma_Dd ** 2) > self.explim):
            return -np.inf
        else:
            llh = math.exp(-((-self.mu + math.log(ddt - self.lam)) ** 2 / (2. * self.sigma ** 2))) / (math.sqrt(2 * math.pi) * (ddt - self.lam) * self.sigma)

            llh_Dd = math.exp(-((-self.mu_Dd + math.log(dd - self.lam_Dd)) ** 2 / (2. * self.sigma_Dd ** 2))) / (math.sqrt(2 * math.pi) * (dd - self.lam_Dd) * self.sigma_Dd)

            if np.isnan(llh) or np.isnan(llh_Dd):
                return -np.inf
            else:
                return np.log(llh) + np.log(llh_Dd)

    def sklogn_analytical_likelihood_Ddonly(self,dd) :
        """
        Evaluates the likelihood of a angular diameter distance Dd(in Mpc) against the model predictions, using a skewed log-normal distribution for dd.
        """
        # skewed lognormal distribution with boundaries
        #if (dd < self.lam_Dd) or ((-self.mu_Dd + math.log(dd - self.lam_Dd)) ** 2 / (2. * self.sigma_Dd ** 2) > self.explim):
         #   return -np.inf
        #else:
        llh_Dd = math.exp(-((-self.mu_Dd + math.log(dd - self.lam_Dd)) ** 2 / (2. * self.sigma_Dd ** 2))) / (math.sqrt(2 * math.pi) * (dd - self.lam_Dd) * self.sigma_Dd)

        if np.isnan(llh_Dd):
            return -np.inf
        else:
            return np.log(llh_Dd)

    def general_likelihood(self,ddt):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a general form.
        It is treating mu as ddt, and sigma as ddt_sigma 
        """
        return -((ddt-self.mu)**2/self.sigma**2)/2
    

    def normal_analytical_likelihood(self, ddt):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a normalised gaussian distribution.
        """
        # Normal distribution with boundaries

        if np.abs(ddt - self.mu) > 3*self.sigma:
            return -np.inf
        else:
            lh = math.exp(- (ddt - self.mu) **2 / (2. * self.sigma **2) ) / (math.sqrt(2 * math.pi) * self.sigma)
            if np.isnan(lh):
                return -np.inf
            else:
                return np.log(lh)


    def kdelikelihood_full(self, kde_kernel, bandwidth):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a loglikelihood sampled from a Kernel Density Estimator. the KDE is constructed using the full ddt samples.

        __ warning:: you should adjust bandwidth to the spacing of your samples chain!
        """
        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(self.ddt_samples, sample_weight=self.weights)
        return kde.score


    def kdelikelihood_hist(self, kde_kernel, bandwidth, nbins_hist):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) against the model predictions, using a loglikelihood sampled from a Kernel Density Estimator. the KDE is constructed using a binned version of the full samples. Greatly improves speed at the cost of a (tiny) loss in precision

        __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

        """
        hist = np.histogram(self.ddt_samples, bins=nbins_hist, weights=self.weights)
        vals = hist[0]
        bins = [(h + hist[1][i+1])/2.0 for i, h in enumerate(hist[1][:-1])]

        # ignore potential zero weights, sklearn does not like them
        kde_bins = [(b,) for v, b in zip(vals, bins) if v>0]
        kde_weights = [v for v in vals if v>0]

        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)
        return kde.score


    def kdelikelihood_hist_2d(self, kde_kernel, bandwidth, nbins_hist):
        """
        Evaluates the likelihood of a angular diameter distance to the deflector Dd (in Mpc) versus its time-delay distance Ddt (in Mpc) against the model predictions, using a loglikelihood sampled from a Kernel Density Estimator. The KDE is constructed using a binned version of the full samples. Greatly improves speed at the cost of a (tiny) loss in precision

        __warning:: you should adjust bandwidth and nbins_hist to the spacing and size of your samples chain!

        __note:: nbins_hist refer to the number of bins per dimension. Hence, the final number of bins will be nbins_hist**2

        """
        hist, dd_edges, ddt_edges = np.histogram2d(x=self.dd_samples, y=self.ddt_samples, bins=nbins_hist, weights=self.weights)
        dd_vals = [(dd + dd_edges[i+1])/2.0 for i, dd in enumerate(dd_edges[:-1])]
        ddt_vals = [(ddt + ddt_edges[i+1])/2.0 for i, ddt in enumerate(ddt_edges[:-1])]

        # ugly but fast enough way to get the correct format for kde estimates
        dd_list, ddt_list, vals = [], [], []
        for idd, dd in enumerate(dd_vals):
            for iddt, ddt in enumerate(ddt_vals):
                dd_list.append(dd)
                ddt_list.append(ddt)
                vals.append(hist[idd, iddt])

        kde_bins = pd.DataFrame.from_dict({"dd": dd_list, "ddt": ddt_list})
        kde_weights = np.array(vals)

        # remove the zero weights values
        kde_bins = kde_bins[kde_weights > 0]
        kde_weights = kde_weights[kde_weights > 0]

        # fit the KDE
        kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth).fit(kde_bins, sample_weight=kde_weights)
        return kde.score


    def hist_lin_interp_likelihood(self, ddt):
        """
        Evaluates the likelihood of a time-delay distance ddt (in Mpc) agains the model predictions, using linear interpolation from an histogram.

        __warning:: for testing purposes only - prefer kdelikelihood_hist, which gives similar results
        """
        if ddt <= self.bins[0] or ddt >= self.bins[-1]:
            return -np.inf
        else:
            indright = np.digitize(ddt, self.bins)
            #return np.log((self.vals[indright-1]+self.vals[indright])/2.0)
            return np.log(self.vals[indright-1])


    def init_loglikelihood(self):
        if self.loglikelihood_type == "sklogn_analytical":
            self.loglikelihood = self.sklogn_analytical_likelihood

        elif self.loglikelihood_type == "sklogn_analytical_Dd":
            self.loglikelihood = self.sklogn_analytical_likelihood_Dd

        elif self.loglikelihood_type == "sklogn_analytical_Ddonly":
            self.loglikelihood = self.sklogn_analytical_likelihood_Ddonly

        elif self.loglikelihood_type == "normal_analytical":
            self.loglikelihood = self.normal_analytical_likelihood

        elif self.loglikelihood_type == "kde_full":
            self.loglikelihood = self.kdelikelihood_full(kde_kernel=self.kde_kernel, bandwidth=self.bandwidth)

        elif self.loglikelihood_type == "kde_hist":
            self.loglikelihood = self.kdelikelihood_hist(kde_kernel=self.kde_kernel, bandwidth=self.bandwidth, nbins_hist=self.nbins_hist)

        elif self.loglikelihood_type == "kde_hist_2d":
            self.loglikelihood = self.kdelikelihood_hist_2d(kde_kernel=self.kde_kernel, bandwidth=self.bandwidth, nbins_hist=self.nbins_hist)

        elif self.loglikelihood_type == "hist_lin_interp":
            self.loglikelihood = self.hist_lin_interp_likelihood
            
        elif self.loglikelihood_type == 'general':
            self.loglikelihood = self.general_likelihood
            
        else:
            assert ValueError("unknown keyword: %s" % self.loglikelihood_type)
            # if you want to implement other likelihood estimators, do it here



class LenstronomyLens(StrongLensSystem):
    """
    This class takes the output of Lenstronomy (Dd versus Ddt distributions) from which it evaluates the likelihood of a Dd versus Ddt (in Mpc) predicted in a given cosmology.

    The default likelihood follows the KDE log-normal distribution implemented in Lenstronomy. You can change the type of kernel used. No other likelihoods have been implemented so far.
    """

    def __init__(self, name, zlens, zsource, ddt_vs_dd_samples, longname=None, loglikelihood_type="kde", kde_type="scipy_gaussian"):
        StrongLensSystem.__init__(self, name=name, zlens=zlens, zsource=zsource, longname=longname)

        self.ddt_vs_dd = ddt_vs_dd_samples     
        self.loglikelihood_type = loglikelihood_type
        self.kde_type = kde_type
        self.init_loglikelihood()

    def kdelikelihood(self, kde_type, bandwidth=20):
        """
        Evaluates the likelihood of a angular diameter distance to the deflector Dd (in Mpc) versus its time-delay distance Ddt (in Mpc) against the model predictions, using a loglikelihood sampled from a Kernel Density Estimator.
        """
        self.ddt = self.ddt_vs_dd["ddt"]
        self.dd = self.ddt_vs_dd["dd"]
        KDEl = KDELikelihood(self.dd.values, self.ddt.values, kde_type=kde_type, bandwidth=bandwidth)
        return KDEl.logLikelihood


    def init_loglikelihood(self):
        if self.loglikelihood_type == "kde_full":
            self.loglikelihood = self.kdelikelihood(kde_type=self.kde_type)
        else:
            assert ValueError("unknown keyword: %s" % self.loglikelihood_type)
            # if you want to implement other likelihood estimators, do it here


"""
The functions below evaluate the joint likelihood function of a set of cosmological parameters in a chosen cosmology against the modeled angular diameter distances of the lens systems.

The core of the process is the sample_params function, that works around the `emcee` MCMC sampler. Provided with uniform priors (log_prior), a list of lens objects and a suitable `astropy.cosmo` cosmology, it computes the angular diameter distances predicted by the cosmology (log_prob_ddt) and evaluate their log-likelihood against the angular diameter distances of the lens systems (log_like_add).

By sampling the cosmological parameters space, sample_params return chains of joint cosmological parameters that can be used to produce fancy posterior plots.
"""

#%%
"""Create the lenses objects"""
dataDir=os.path.dirname(os.path.abspath(__file__))+'/'

"""B1608"""
# B1608 using Ddt only, first analysis from Suyu+2010
B1608 = GLEELens(name="B1608", longname="B1608 (Suyu+2010)", zlens=0.6304, zsource=1.394,
                    mu=7.0531390, sigma=0.2282395, lam=4000.0,
                    loglikelihood_type="sklogn_analytical"
                   )

# B1608 using Dd only, analysis from Jee+2019
B1608_Ddonly = GLEELens(name="B1608Dd", longname="B1608 Dd (Jee+2019)", zlens=0.6304, zsource=1.394,
                    mu=7.0531390, sigma=0.2282395, lam=4000.0, mu_Dd = 6.79671, sigma_Dd=0.1836, lam_Dd = 334.2,
                    loglikelihood_type="sklogn_analytical_Ddonly"
                   )
# B1608 using both Ddt and Dd only, analysis from Suyu+2010, Jee+2019, used in Wong+2019
B1608_DdDdt = GLEELens(name="B1608DdDdt", longname="B1608 (Suyu+2010, Jee+2019)", zlens=0.6304, zsource=1.394,
                    mu=7.0531390, sigma=0.2282395, lam=4000.0, mu_Dd = 6.79671, sigma_Dd=0.1836, lam_Dd = 334.2,
                    loglikelihood_type="sklogn_analytical_Dd"
                   )


"""J1206"""
ddt_vs_dd_1206s = pd.read_csv(dataDir+"h0licow_distance_chains/J1206_final.csv")
J1206 = LenstronomyLens(name="J1206", longname="J1206 (Birrer+2019)", zlens=0.745, zsource=1.789, 
                           ddt_vs_dd_samples=ddt_vs_dd_1206s,
                           loglikelihood_type="kde_full", kde_type="scipy_gaussian",
                          )


"""WFI2033"""
#preprocess the 2033 chains...
ddt_2033s_bic = pd.read_csv(dataDir+"h0licow_distance_chains/wfi2033_dt_bic.dat")

# remove the Ddt that are above 8000, as it makes it hard to have a decent kde fit.
cutweights = [w for w, ddt in zip(ddt_2033s_bic["weight"], ddt_2033s_bic["Dt"]) if 0 < ddt < 8000]
cutddts = [ddt for ddt in ddt_2033s_bic["Dt"] if 0 < ddt < 8000] 
ddt_2033s_bic = pd.DataFrame.from_dict(data={"ddt": cutddts, "weight": cutweights})

# create the lens object
WFI2033 = GLEELens(name="WFI2033", longname="WFI2033 (Rusu+2019)", zlens=0.6575, zsource=1.662,
                      loglikelihood_type="kde_hist", kde_kernel="gaussian", ddt_samples=ddt_2033s_bic['ddt'],
                          weights=ddt_2033s_bic['weight'],
                          bandwidth=20, nbins_hist=400
                     )



"""HE0435"""
#Using only HST data, analysis from Wong+2017
HE0435_HST = GLEELens(name="HE0435_HST", longname="HE0435-HST (Wong+2017)", zlens=0.4546, zsource=1.693,
                     mu=7.57930024e+00, sigma=1.03124167e-01, lam=6.53901645e+02,
                     loglikelihood_type="sklogn_analytical"
                    )
                

#Using HST + AO data, analysis from Chen+2019
ddt_0435s_AO_HST = pd.read_csv(dataDir+"h0licow_distance_chains/HE0435_Ddt_AO+HST.dat", delimiter=" ", skiprows=1, names=("ddt",))
ddt_0435s_AO_HST["weight"] = np.ones(len(ddt_0435s_AO_HST["ddt"]))

HE0435_AO_HST = GLEELens(name="HE0435_AO_HST", longname="HE0435 (Wong+2017, Chen+2019)", zlens=0.4546, zsource=1.693,
                    loglikelihood_type="kde_hist", kde_kernel="gaussian", ddt_samples=ddt_0435s_AO_HST['ddt'],
                        weights = ddt_0435s_AO_HST["weight"], 
                      bandwidth=20, nbins_hist=400
                    )


"""RXJ1131"""
#Using only HST data, analysis from Suyu+2014
RXJ1131_HST = GLEELens(name="RXJ1131_HST", longname="RXJ1131-HST (Suyu+2014)", zlens=0.295, zsource=0.654, 
                      mu=6.4682, sigma=0.20560, lam=1388.8, 
                      loglikelihood_type="sklogn_analytical"
                     )


#Using HST + AO data, analysis from Chen+2019
dd_vs_ddt_1131s_AO_HST = pd.read_csv(dataDir+"h0licow_distance_chains/RXJ1131_AO+HST_Dd_Ddt.dat", 
                                  delimiter=" ", skiprows=1, names=("dd", "ddt"))
RXJ1131_AO_HST = GLEELens(name="RXJ1131_AO_HST", longname="RXJ1131 (Suyu+2014, Chen+2019)", 
                              zlens=0.295, zsource=0.654,
                              loglikelihood_type="kde_hist_2d", kde_kernel="gaussian",
                              bandwidth=20, nbins_hist=80,
                              ddt_samples=dd_vs_ddt_1131s_AO_HST["ddt"], dd_samples=dd_vs_ddt_1131s_AO_HST["dd"]
                     )

"""PG1115"""
#Using HST + AO data, analysis from Chen+2019
dd_vs_ddt_1115s= pd.read_csv(dataDir+"h0licow_distance_chains/PG1115_AO+HST_Dd_Ddt.dat", 
                                  delimiter=" ", skiprows=1, names=("dd", "ddt"))
PG1115 = GLEELens(name="PG1115", longname="PG1115 (Chen+2019)", 
                              zlens=0.311, zsource=1.722,
                              loglikelihood_type="kde_hist_2d", kde_kernel="gaussian",
                              bandwidth=20, nbins_hist=80,
                              ddt_samples=dd_vs_ddt_1115s["ddt"], dd_samples=dd_vs_ddt_1115s["dd"]
                     )

#%%

ddt_DES0408 = pd.read_csv(dataDir+"h0licow_distance_chains/DES0408-5354/power_law_dist_post_no_kext.txt", delimiter=" ", names=("ddt",))
ddt_DES0408["weight"] = np.ones(len(ddt_DES0408["ddt"]))

DES0408 = GLEELens(name="DES0408-5354", longname="DES0408-5354", zlens=0.597, zsource=2.375,
                      loglikelihood_type="kde_hist", kde_kernel="gaussian", ddt_samples=ddt_DES0408["ddt"],
                          bandwidth=20, nbins_hist=100
                     )

DES0408_Ddt = GLEELens(name="DES0408-5354", longname="DES0408-5354", zlens=0.597, zsource=2.375, mu=3382, sigma=130.5,
                       loglikelihood_type="general")

#%%
lenses = [B1608_DdDdt, RXJ1131_AO_HST, HE0435_AO_HST, J1206, WFI2033, PG1115,DES0408_Ddt]
# lenses = [DES0408_Ddt]
#%%

class TD(object):
    def __init__(self):
        self.lenses=lenses
        
    def log_like_add(self,lens,cosmo):
        """
        Computes the relevant angular diameter distance(s) of a given lens in a given cosmology,
        and evaluate its/their joint likelihood against the same modeled distances of the lens.
    
        param lens: either a GLEELens or LenstronomyLens instance.
        param cosmo: an astropy cosmology object. 
        """
        dd = cosmo.ang_dis_z(lens.zlens)
        ds = cosmo.ang_dis_z(lens.zsource)
        dds = cosmo.ang_dis_z2(lens.zlens, lens.zsource)
        ddt = (1. + lens.zlens) * dd * ds / dds
    
        if isinstance(lens, GLEELens):
            if lens.loglikelihood_type in ["kde_full", "kde_hist"]:
                # because the newest sklearn kde want arrays, not numbers... 
                return lens.loglikelihood(np.array(ddt).reshape(1, -1))
            elif lens.loglikelihood_type in ["kde_hist_2d"]:
                return lens.loglikelihood(np.array([dd, ddt]).reshape(1, -1))
            elif lens.loglikelihood_type in ["sklogn_analytical_Dd"] :
                return lens.loglikelihood(ddt, dd)
            elif lens.loglikelihood_type in ["sklogn_analytical_Ddonly"] :
                return lens.loglikelihood(dd)
            else:
                return lens.loglikelihood(ddt)
    
        elif isinstance(lens, LenstronomyLens):
            return lens.loglikelihood(dd, ddt)
    
        else:
            sys.exit("I don't know what to do with %s, unknown instance" % lens)
    
    
    def log_prob_ddt(self,cosmo):
        """
        Compute the likelihood of the given cosmological parameters against the
        modeled angular diameter distances of the lenses.
    
        param theta: list of loat, folded cosmological parameters.
        param lenses: list of lens objects (currently either GLEELens or LenstronomyLens).
        param cosmology: string, keyword indicating the choice of cosmology to work with.
        """
        
        logprob = 0
        for lens in lenses:
            logprob += self.log_like_add(lens=lens,cosmo=cosmo)
        return logprob
    
    def chi2(self,cosmo):
        return -self.log_prob_ddt(cosmo)*2
    
    
    # def sample_params(self, nwalkers=32, nsamples=20000, save=True, filepath="temp.pkl", cluster = False):
    #     """
    #     High-level wrapper around the above functions. Explore the cosmological parameters space and
    #     return their likelihood evaluated against the modeled angular diameter distances
    #     of (multiple) lens system(s).
    
    #     param lenses: list of lens objects (currently either GLEELens or LenstronomyLens).
    #     param cosmology: string, keyword indicating the choice of cosmology to work with.
    #     param nwalkers: int, number of emcee walkers used to sample the parameters space.
    #     param nsamples: int, number of samples for an MCMC chain to converge. 
    #         Make sure these are larger than the autocorrelation time!
    #     param save: boolean, if True the combined, flattened chain is saved in filepath
    #     param filepath: string, path of where the output chain is saved.
    #     """
    
    
    #     # Our starting point is a "decent" solution. You might want to check it does not impact the results, but unless you start from really crazy values, it shouldn't. We slightly randomize the starting point for each walker.
    #     startpos = self.params_all['fit']  # H0, Om, Ok
    
    #     pos = startpos + 1e-4*np.random.randn(nwalkers, len(startpos))
    #     nwalkers, ndim = pos.shape
    
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob_ddt)
    #     if cluster :
    #         sampler.run_mcmc(pos, nsamples)
    #     else :
    #         sampler.run_mcmc(pos, nsamples, progress =True)
    
    #     self.savefile(sampler,nwalkers)
    #     return sampler

    # def savefile(self,sampler,nwalkers):
    #     chains_dir='./chains/'
    #     if not os.path.exists(chains_dir):
    #         os.makedirs(chains_dir)
        
    #     savefile_name='./chains/'+self.Chains_name+'.npy'
    #     burnin = 100
    #     samples = sampler.chain[:, burnin:, :].reshape((-1, self.n))
    #     self.samples=samples
    #     vv=lambda v: (v[1], v[2]-v[1], v[1]-v[0])
    #     self.theta_fact= vv(np.percentile(samples, [16, 50, 84],axis=0))
    #     self.minkaf=0
        
    #     ranges=list(zip(self.params_all['lower'],self.params_all['upper']))
        
    #     np.save(savefile_name,(samples,self.params_all['name'],self.params_all['fit'],self.theta_fact,self.minkaf,0,ranges))
    #     print('\nChains name is "%s".'%self.Chains_name)





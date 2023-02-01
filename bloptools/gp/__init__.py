import torch
import gpytorch
import numpy as np
import scipy as sp

import pandas as pd

import warnings

import bluesky.plans as bp  # noqa F401
import databroker

import time as ttime

from scipy.stats import qmc




from . import kernels, utils

class GaussianProcessRegressor(gpytorch.models.ExactGP):
    
    def __init__(self, x, y, likelihood, n_dof, batch_shape=1):
        super().__init__(x, y, likelihood)
        
        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=batch_shape)        
        self.covar_module = kernels.LatentMaternKernel(n_dof=n_dof, off_diag=True) 

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class GPRC():
    '''
    A Gaussian Process Regressor and Classifier model, with learning methods.
    '''
    
    def __init__(self, **kwargs):

        pass
        
    def set_data(self, x, y):
        '''
        Set the data with parameters and values.
        
        x: parameters
        y: function values at those parameters
        
        Passed parameters must be between [-1, 1] in every dimension. Passed values need not be finite. 
        '''
        
        if (x.min(axis=0) <= -1).any() or (x.max(axis=0) >= +1).any():
            raise ValueError('Parameters must be between -1 and +1 in each dimension.')
        
        self.x, self.y = np.atleast_2d(x), np.atleast_1d(y)
        self.n, self.n_dof = self.x.shape
        
        # g is a boolean describing whether or not the data can be regressed upon
        self.g = (~np.isnan(self.y)) & (np.isfinite(self.y))
        
        # nu is the maximum value of the optimum
        self.nu = np.nanmax(self.y)
        
        # prepare Gaussian process ingredients for the regressor and classifier 
        # use only regressable points for the regressor
        self.regressor_inputs   = torch.as_tensor(self.x[self.g]).float() 
        self.regressor_targets  = torch.as_tensor(self.y[self.g]).float()
        
        self.classifier_inputs  = torch.as_tensor(self.x).float()
        self.classifier_targets = torch.as_tensor(self.g.astype(int))
        
        self.regressor_likelihood  = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        self.regressor_likelihood  = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, 1e-2 * self.y[self.g].std()))
        self.classifier_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(self.classifier_targets, learn_additional_noise=True)
        
        
    def train(self, training_iter=100, hypers=None, verbose=True):
    
        # instantiate regressor
        self.regressor =  GaussianProcessRegressor(self.regressor_inputs, 
                                                   self.regressor_targets, 
                                                   self.regressor_likelihood,
                                                   self.n_dof,
                                                  )
        
        # instantiate classifier. 
        # the batch_shape corresponds to our two possible classes, "good" and "bad"
        self.classifier = GaussianProcessRegressor(self.classifier_inputs, 
                                                   self.classifier_likelihood.transformed_targets, 
                                                   self.classifier_likelihood, 
                                                   self.n_dof,
                                                   batch_shape=2)
        
        # if hyperparameters are passed, specify 
        if not hypers is None: 
            self.regressor.initialize(**hypers)
            self.classifier.initialize(**hypers)
    
        self.regressor_likelihood.train()
        self.regressor.train()
        
        self.classifier_likelihood.train()
        self.classifier.train()

        # Use the adam optimizer
        self.regressor_optimizer  = torch.optim.Adam(self.regressor.parameters(), lr=1e-1)
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        self.regressor_mll  = gpytorch.mlls.ExactMarginalLogLikelihood(self.regressor_likelihood, self.regressor)
        self.classifier_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.classifier_likelihood, self.classifier)

        for i in range(training_iter):
            
            # torch.any(torch.isnan(v))
            
            self.regressor_optimizer.zero_grad()
            loss = - self.regressor_mll(self.regressor(self.regressor_inputs), self.regressor_targets)
            loss.backward()
            self.regressor_optimizer.step()
            
            self.classifier_optimizer.zero_grad()
            loss = - self.classifier_mll(self.classifier(self.classifier_inputs), self.classifier_likelihood.transformed_targets).sum()
            loss.backward()
            self.classifier_optimizer.step()
            
            if (i + 1) % 100 == 0:
                if verbose:
                    print(f'iter {i + 1}',
                          f'scale: {self.regressor.covar_module.outputscale.detach().numpy().ravel().round(3)}',
                          f'diagonal: {self.regressor.covar_module.trans_diagonal.detach().numpy().ravel().round(3)}',
                         )
            

    def update(self, x, data, reuse_hypers):
                
        self.set_data(np.r_[self.x, x], pd.concat([self.data, data]))
        hypers = self.hypers if reuse_hypers else None
        self.train(training_iter=300, hypers=hypers, verbose=False)
        
    def regress(self, x):
        
        x = torch.as_tensor(np.atleast_2d(x)).float()
        
        # set to evaluation mode
        self.regressor_likelihood.eval()
        self.regressor.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            prediction = self.regressor_likelihood(self.regressor(x))
            
        return prediction
    
    def classify(self, x, return_variance=False):
        
        x = torch.as_tensor(np.atleast_2d(x)).float()
        
        # set to evaluation mode
        self.classifier_likelihood.eval()
        self.classifier.eval()
    
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            
            dist = self.classifier(x)
            samples = dist.sample(torch.Size((256,))).exp()
            probabilities = (samples / samples.sum(-2, keepdim=True)).mean(0)
            
        if return_variance: 
            res = probabilities[1].detach().numpy(), dist.variance.detach().numpy()
        else:
            res = probabilities[1].detach().numpy()
            
        return res
    
    def precision(self, x):
        
        return 1 / self.sigma(x)
    
    @property
    def confidence_bound(self, x, z):
        
        prediction = self.regress(x)
        
        return (prediction.mean + z * prediction.stddev).detach().numpy()
    
    def mean(self, x):
                
        return self.regress(x).mean.detach().numpy().ravel()
    
    def sigma(self, x):
        
        return self.regress(x).stddev.detach().numpy().ravel()

#self.trans_fun_y = np.interp(x, y, 

class Optimizer():
    
    def __init__(self, 
                 watchpoint,
                 dofs,
                 dof_bounds, 
                 run_engine,
                 db,
                 init_params=None, 
                 init_data=None,
                 init_scheme=None, 
                 fitness_mode='density',
                 n_init=None, 
                 init_training_iter=1000, 
                 verbose=True, 
                 **kwargs):
        
        self.dof_bounds = dof_bounds

        self.run_engine = run_engine
        self.watchpoint = watchpoint
        self.n_dof = len(dofs)
        self.fitness_mode = 'density'
        self.dofs = dofs
        self.db = db
        
        # for actual prediction and optimization
        self.model = GPRC()
        
        # for shenanigans
        self.dummy_model = GPRC() 
        
        
        if (init_params is not None) and (init_data is not None):
            
            self.params, self.data = init_params, init_data
            
        elif init_scheme == 'quasi-random': 
            
            n_init = n_init if n_init is not None else 3 ** self.n_dof
            self.autoinitialize(n=n_init, scheme='quasi-random', verbose=verbose)
        
        else: raise Exception('Could not initialize model!')
        
        # convert params to x 
        self.params_trans_fun = lambda params : 2 * (params - self.dof_bounds.min(axis=0)) / self.dof_bounds.ptp(axis=0) - 1
        
        # convert x to params
        self.inv_params_trans_fun = lambda x : 0.5 * (x + 1) * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)
            
        self.x = self.params_trans_fun(self.params)
        
        self.compute_fitness()
        self.y = self.fitness_trans_fun(self.fitness)

        self.model.set_data(self.x, self.y)     
        self.model.train(training_iter=init_training_iter, verbose=verbose)
                
            
        
        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.test_params = sampler.random(n=1024) * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)
        self.test_params = self.test_params[self.test_params[:,0].argsort()]
        
        # a more lightweight proxy for Quasi-Monte Carlo integration
        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        self.qmc_params = sampler.random(n=64) * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)
        self.qmc_params = self.qmc_params[self.qmc_params[:,0].argsort()]
        
    def compute_fitness(self):
        
        if self.fitness_mode == 'density':
        
            data_cols = [f'{self.watchpoint.name}_image', 
                         f'{self.watchpoint.name}_horizontal_extent', 
                         f'{self.watchpoint.name}_vertical_extent']
            
            self.fitness = np.array([utils.get_density(sp.ndimage.zoom(image, zoom=2), (x_extent, y_extent))
                                     for image, x_extent, y_extent in self.data[data_cols].values])
        
            # convert fitness to y
            self.fitness_trans_fun = lambda fitness : np.log(fitness)
            
            # convert y to fitness
            self.inv_fitness_trans_fun = lambda y : np.exp(y) 
    
    def autoinitialize(self, n, verbose, scheme='quasi-random'):
        
        halton_sampler = qmc.Halton(d=self.n_dof, scramble=True)
        
        self.params = halton_sampler.random(n=2**int(np.log(n)/np.log(2)+1))[:n] * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)  
        self.data   = self.acquire(self.params, verbose=verbose)
        
        
    def update(self, new_params, new_data, reuse_hypers):
                
        self.params = np.r_[self.params, new_params]   
        self.data = pd.concat([self.data, new_data])
        
        self.compute_fitness()
        
        self.x = self.params_trans_fun(self.params)
        self.y = self.fitness_trans_fun(self.fitness)
        
        self.model.set_data(self.x, self.y)
        hypers = self.hypers if reuse_hypers else None
        self.model.train(training_iter=300, hypers=hypers, verbose=False)
        
    def acquire(self, params, verbose=False): 
        
        table = pd.DataFrame(columns=['daq_time', 'acq_log'])
        
        for _params in params:
            
            if verbose: print(f'sampling {_params}')
            start_time = ttime.monotonic()

            try:

                (uid,) = self.run_engine(bp.list_scan([self.watchpoint], *[_ for items in zip(self.dofs, np.atleast_2d(_params).T) for _ in items]))
                _table = self.db[uid].table(fill=True)
                _table.insert(0, 'daq_time', ttime.monotonic() - start_time)
                _table.insert(1, 'acq_log', 'ok')
                
            except Exception as err:

                warnings.warn(err.args[0])
                _table = pd.DataFrame([(ttime.monotonic() - start_time, err.args[0])], columns=['daq_time', 'acq_log'])
                
            table = pd.concat([table, _table])
                          
        return table
    
    def learn(self, strategy, n_iter=1, n_per_iter=1, options={}, reuse_hypers=True, verbose=True):
        
        #ip.display.clear_output(wait=True)
        print(f'learning with strategy "{strategy}" ...')
                          
        for i in range(n_iter):

            params_to_sample = np.atleast_2d(self.recommend(n=n_per_iter, strategy=strategy)) # get point(s) to sample from the strategizer
            
            res_table = self.acquire(params_to_sample) # sample the point(s)
            
            self.update(new_params=params_to_sample, new_data=res_table, reuse_hypers=reuse_hypers) # update our model 
            
            if verbose: print(f'# {i+1:>03} : {params_to_sample.round(4)} -> {self.fitness[-1]:.04e}')
        
    
    def _beam_density(self, res): return utils.get_density(res)
    
    def _get_min_lcb(self, z):
        
        x0 = self.test_params[self.confidence_bound(self.test_params, z).argmin()].detach().numpy()
        opt_res = sp.optimize.minimize(self.confidence_bound, x0=x0, args=(z,), bounds=self.dof_bounds.T, method='SLSQP')
        
        return opt_res.x
    
    def _get_min_lcb(self, z):
        
        x0 = self.test_params[self.confidence_bound(self.test_params, z).argmin()]
        opt_res = sp.optimize.minimize(self.confidence_bound, x0=x0, args=(z,), bounds=self.dof_bounds.T, method='SLSQP')
        
        return opt_res.x
    
    @property
    def hypers(self): return {
                    'outputscale' : self.regressor.covar_module.outputscale,
                   'trans_matrix' : self.regressor.covar_module.trans_matrix
                 }


    ### MEI strategy
    
    @property
    def optimum(self): 
        return self.inv_fitness_trans_fun(self.model.x[np.nanargmax(self.model.y)])

    
    def _negative_expected_information_gain(self, params):
        return -self.entropy(params)

    def _negative_improvement_variance(self, params):
        
        x = self.params_trans_fun(params)
        
        mu    = self.model.mean(x)
        sigma = self.model.sigma(x)
        nu    = self.model.nu
        p     = self.model.classify(x)
        
        # sigma += 1e-3 * np.random.uniform(size=sigma.shape)
        
        A = np.exp(-0.5 * np.square((mu - nu)/sigma)) / (np.sqrt(2*np.pi) * sigma)
        B = 0.5 * (1 + sp.special.erf((mu - nu)/(np.sqrt(2) * sigma)))
        
        V = - p ** 2 * (A * sigma ** 2 + B * (mu - nu)) ** 2 + p * (A * sigma ** 2 * (mu - nu) + B * (sigma ** 2 + (mu - nu) ** 2))

        return - np.maximum(0, V)
    
    
    
    def recommend(self, n=1, strategy=None):
        '''
        Recommends the next $n$ points to sample, according to the given strategy. 
        '''
        
        sampler = sp.stats.qmc.Halton(d=self.n_dof, scramble=True)
        iter_test_params = (sampler.random(n=256*n) * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)).reshape(-1, n, self.n_dof)

        if strategy == 'maximize_expected_improvement': 
            
            # pass the whole array, but reshape 
            return self._argmax_expected_improvement(iter_test_params)
        
        
        if strategy == 'maximize_expected_information': 
            
            # pass the whole array, but reshape 
            return self._argmax_expected_information(iter_test_params)
        
        
    # talk to the model
        
    def mean(self, params):     
        return self.inv_fitness_trans_fun(self.model.mean(self.params_trans_fun(params).reshape(-1,self.n_dof))).reshape(params.shape[:-1])
    
    def sigma(self, params):     
        return self.inv_fitness_trans_fun(self.model.sigma(self.params_trans_fun(params).reshape(-1,self.n_dof))).reshape(params.shape[:-1])
    
    def classify(self, params):     
        return self.model.classify(self.params_trans_fun(params).reshape(-1,self.n_dof)).reshape(params.shape[:-1])
    
    def entropy(self, params):
        return np.log(np.sqrt(2*np.pi*np.e)*self.sigma(params))
        
    # functions for expected_improvement strategies
        
    def _argmax_expected_improvement(self, PARAMS):
        '''
        From an array with shape (n_sets, n_params_per_set, n_dof), return the (n_sets,) expected improvements. 
        Assumes independent improvements. 
        '''
        
        x0 = PARAMS[np.argmin(self._negative_expected_improvement(PARAMS).sum(axis=1))]
        
        return x0
    
    def _negative_expected_improvement(self, params):
        '''
        Returns the negative expected improvement over the maximum, in GP units. 
        '''
        
        x = self.params_trans_fun(params).reshape(-1, self.n_dof)
        
        # using GPRC units here
        mu    = self.model.mean(x)
        sigma = self.model.sigma(x)
        nu    = self.model.nu
        p     = self.model.classify(x)
        
        A = np.exp(-0.5 * np.square((mu - nu)/sigma)) / (np.sqrt(2*np.pi) * sigma)
        B = 0.5 * (1 + sp.special.erf((mu - nu)/(np.sqrt(2) * sigma)))
        E = - p * (A * sigma ** 2 + B * (mu - nu))

        return E.reshape(params.shape[:-1])
    
    
    
    # functions for expected_information strategies
    
    
    
    
    def _argmax_expected_information(self, PARAMS):
        
        # pass it as a 2D array, but reshape the result
        x0 = PARAMS[np.argmin(self._negative_expected_information(PARAMS))]
        
        return x0
            
    
    def _negative_expected_information(self, params):
        
        current_info = -self._posterior_entropy(params=None)
        potential_info = -self._posterior_entropy(params=params)
        p_valid = self.classify(params)
        
        return - np.product(p_valid, axis=-1) * (potential_info - current_info) 
    
    
    
    
    
    
        
#     def _get_maximum_posterior_entropy(self, **kwargs):
        
#         n_per_iter
        
#         sampler = sp.stats.qmc.Halton(d=n_dof, scramble=True)
        
        
#         if scheme in functions_to_minimize.keys():
#             fun = functions_to_minimize[scheme]
#         else: 
#             raise ValueError(f'scheme "{scheme}" is not a valid scheme. schemes must be one of:\n{functions_to_minimize.keys()}')
        
#         sampler = sp.stats.qmc.Halton(d=n_dof, scramble=True)
#         test_params  = sampler.random(n=1024) * self.dof_bounds.ptp(axis=0) + self.dof_bounds.min(axis=0)
        
#         x0 = test_params[fun(test_params).argmin()]
#         opt_res = sp.optimize.minimize(fun, x0=x0, bounds=self.dof_bounds.T, method='SLSQP', options={'maxiter':1024})
    
#         return x0

    def _posterior_entropy(self, params=None):
        
        '''
        params is an array with shape (n_sets, n_params_per_set, n_dof)
        
        If we observe each of the n_params_per_set in each of the n_sets, what will the resulting integrals over the posterior rate be?
        This function estimates that using a Quasi-Monte Carlo integration over a dummy Gaussian processes.
        Returns an array of shape (n_sets,).
        
        If None is passed, we return the posterior entropy of the real process. 
        '''
        
        if params is None: 
            params = np.empty((1, 0, self.n_dof)) # one set of zero observations
    
        # get the noise from the regressor likelihood
        raw_noise = self.model.regressor.state_dict()['likelihood.noise_covar.raw_noise']
        noise = self.model.regressor.likelihood.noise_covar.raw_noise_constraint.transform(raw_noise).item()
        
        # x_data is an array of shape (n_sets, n_process + n_params_per_set, n_dof) that describes potential obervation states
        # x_star is an array of points at which to evaluate the entropy rate, to sum together for the QMCI
        x_data = torch.as_tensor(np.r_[[np.r_[self.model.x[self.model.g], _x] for _x in np.atleast_3d(self.params_trans_fun(params))]])
        x_star = torch.as_tensor(self.params_trans_fun(self.test_params))

        # for each potential observation state, compute the prior-prior and prior-posterior covariance matrices
        K_data_data = self.model.regressor.covar_module(x_data, x_data).detach().numpy().astype(float) + noise ** 2 * np.eye(x_data.shape[1])
        K_star_data = self.model.regressor.covar_module(x_star, x_data).detach().numpy().astype(float)

        prior_variance = self.model.regressor.covar_module.outputscale.item() + noise ** 2

        # sum over eigenvalues to get the diagonal of the weighted outer product. a very cheek identity!
        posterior_variance = prior_variance - (np.matmul(K_star_data, np.linalg.inv(K_data_data)) * K_star_data).sum(axis=-1)

        marginal_entropy_rate = 0.5*np.log(2*np.pi*np.e*posterior_variance)

        return marginal_entropy_rate.sum(axis=-1)

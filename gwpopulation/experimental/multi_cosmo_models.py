"""
Cosmological functionality in :code:`GWPopulation` is based on the :code:`wcosmo` package.
For more details see the `wcosmo documentation <https://wcosmo.readthedocs.io/en/latest/>`_.

We provide a mixin class :func:`gwpopulation.experimental.cosmo_models.CosmoMixin` that
can be used to add cosmological functionality to a population model.
"""

import numpy as xp
from wcosmo import z_at_value
from wcosmo.astropy import WCosmoMixin, available
from wcosmo.utils import disable_units as wcosmo_disable_units

from .jax import NonCachingModel


class TwoCosmoMixin:
    """
    Mixin class that provides cosmological functionality to a subclass.

    Parameters
    ==========
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, cosmo_model1="Planck15", cosmo_model2="Planck15"):
        wcosmo_disable_units()
        self.cosmo_model1 = cosmo_model1
        if self.cosmo_model1 == "FlatwCDM":
            self.cosmology_names1 = ["H01", "Om01", "w01"]
        elif self.cosmo_model1 == "FlatLambdaCDM":
            self.cosmology_names1 = ["H01", "Om01"]
        else:
            self.cosmology_names1 = []
        self._cosmo1 = available[cosmo_model1]

        self.cosmo_model2 = cosmo_model2
        if self.cosmo_model2 == "FlatwCDM":
            self.cosmology_names2 = ["H02", "Om02", "w02"]
        elif self.cosmo_model2 == "FlatLambdaCDM":
            self.cosmology_names2 = ["H02", "Om02"]
        else:
            self.cosmology_names2 = []
        self._cosmo2 = available[cosmo_model2]

    def cosmology_variables1(self, parameters):
        """
        Extract the cosmological parameters from the provided parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        dict
            A dictionary containing :code:`self.cosmology_names` as keys.
        """
        return {key: parameters[key] for key in self.cosmology_names1}

    def cosmology_variables2(self, parameters):
        """
        Extract the cosmological parameters from the provided parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        dict
            A dictionary containing :code:`self.cosmology_names` as keys.
        """
        return {key: parameters[key] for key in self.cosmology_names2}

    def cosmology1(self, parameters):
        """
        Return the cosmology model given the parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        wcosmo.astropy.WCosmoMixin
            The cosmology model.
        """
        if isinstance(self._cosmo1, WCosmoMixin):
            return self._cosmo1
        else:
            return self._cosmo1(**self.cosmology_variables1(parameters))
    
    def cosmology2(self, parameters):
        """
        Return the cosmology model given the parameters.

        Parameters
        ==========
        parameters: dict
            The parameters for the cosmology model.

        Returns
        =======
        wcosmo.astropy.WCosmoMixin
            The cosmology model.
        """
        if isinstance(self._cosmo2, WCosmoMixin):
            return self._cosmo2
        else:
            return self._cosmo2(**self.cosmology_variables2(parameters))

    def detector_frame_to_source_frame(self, data, **parameters):
        r"""
        Convert detector frame samples to sourece frame samples given cosmological
        parameters. Calculate the corresponding
        :math:`\frac{d \theta_{\rm detector}}{d \theta_{\rm source}}` Jacobian term.
        This includes factors of :math:`(1 + z)` for redshifted quantities.

        Parameters
        ==========
        data: dict
            Dictionary containing the samples in detector frame.
        parameters: dict
            The cosmological parameters for relevant cosmology model.

        Returns
        =======
        samples: dict
            Dictionary containing the samples in source frame.
        jacobian: array-like
            The Jacobian term.
        """

        samples = dict()
        if "luminosity_distance" in data.keys():
            cosmo1 = self.cosmology1(self.parameters)
            samples1["redshift"] = z_at_value(
                cosmo1.luminosity_distance,
                data["luminosity_distance"],
            )
            jacobian1 = cosmo1.dDLdz(samples1["redshift"])

            cosmo2 = self.cosmology2(self.parameters)
            samples2["redshift"] = z_at_value(
                cosmo2.luminosity_distance,
                data["luminosity_distance"],
            )
            jacobian2 = cosmo1.dDLdz(samples2["redshift"])
        elif "redshift" not in data:
            raise ValueError(
                f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
            )
        else:
            raise ValueError(
                f"There has to be luminosity distance!"
            )

        for key in list(data.keys()):
            if key.endswith("_detector"):
                samples1[key[:-9]] = data[key] / (1 + samples1["redshift"])
                jacobian1 *= 1 + samples1["redshift"]

                samples2[key[:-9]] = data[key] / (1 + samples2["redshift"])
                jacobian2 *= 1 + samples2["redshift"]
            elif key != "luminosity_distance":
                samples1[key] = data[key]
                samples2[key] = data[key]

        return samples1, jacobian1, samples2, jacobian2

class TwoNonCachingModel:
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """
    r"""
    Population model that combines a set of factorizable models.

    This should take population parameters and return the probability.

    .. math::

        p(\theta | \Lambda) = \prod_{i} p_{i}(\theta | \Lambda)
    """

    def __init__(self, model_functions1=None, model_functions2=None):
        """
        Parameters
        ==========
        model_functions: list
            List of callables to compute the probability.
            If this includes classes, the `__call__` method should return the
            probability.
            The requires variables are chosen at run time based on either
            inspection or querying a :code:`variable_names` attribute.
        """
        self.models1 = model_functions1
        self.models2 = model_functions2
        self.parameters = dict()

    def _get_function_parameters(self, func):
        """
        If the function is a class method we need to remove more arguments or
        have the variable names provided in the class.
        """
        if hasattr(func, "variable_names"):
            param_keys = func.variable_names
        else:
            param_keys = infer_args_from_function_except_n_args(func, n=0)
            ignore = ["dataset", "data", "self", "cls"]
            for key in ignore:
                if key in param_keys:
                    del param_keys[param_keys.index(key)]
        parameters = {key: self.parameters[key] for key in param_keys}
        return parameters

    def prob(self, data1, data2, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        Parameters
        ==========
        data: dict
            Dictionary containing the points at which to evaluate the
            population model.
        kwargs: dict
            The population parameters. These cannot include any of
            :code:`["dataset", "data", "self", "cls"]` unless the
            :code:`variable_names` attribute is available for the relevant
            model.
        """
        probability1 = 1.0
        for function in self.models1:
            new_probability = function(data1, **self._get_function_parameters(function))
            probability1 *= new_probability

        probability2 = 1.0
        for function in self.models2:
            new_probability = function(data2, **self._get_function_parameters(function))
            probability2 *= new_probability
        return probability1, probability2

class TwoCosmoModel(TwoNonCachingModel, TwoCosmoMixin):
    """
    Modified version of :code:`bilby.hyper.model.Model` that automatically
    updates the source-frame quantities given the detector-frame quantities and
    cosmology and disables caching due to the source-frame quantities changing
    every iteration.

    Parameters
    ==========
    model_functions: list
        List containing the model functions.
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, model_functions1=None, model_functions2=None, cosmo_model1="Planck15", cosmo_model2="Planck15"):
        TwoNonCachingModel.__init__(self, model_functions1=model_functions1, model_functions2=model_functions2)
        TwoCosmoMixin.__init__(self, cosmo_model1=cosmo_model1, cosmo_model2=cosmo_model2)

    def prob(self, data, **kwargs):
        """
        Compute the total population probability for the provided data given
        the keyword arguments.

        This method augments :code:`bilby.hyper.model.Model.prob` by converting
        the detector frame samples to source frame samples and dividing by the
        corresponding Jacobian term.

        Parameters
        ==========
        data: dict
            Dictionary containing the points at which to evaluate the
            population model.
        kwargs: dict
            The population parameters. These cannot include any of
            :code:`["dataset", "data", "self", "cls"]` unless the
            :code:`variable_names` attribute is available for the relevant
            model.
        """

        data1, jacobian1, data2, jacobian2 = self.detector_frame_to_source_frame(data)
        probability1, probability2 = super().prob(data1, data2, **kwargs)
        probability1 /= jacobian1
        probability2 /= jacobian2

        return self.parameters['likelihood_mix']*probability1 + (1-self.parameters['likelihood_mix'])*probability2

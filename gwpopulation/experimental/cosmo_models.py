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


class CosmoMixin:
    """
    Mixin class that provides cosmological functionality to a subclass.

    Parameters
    ==========
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, cosmo_model="Planck15"):
        wcosmo_disable_units()
        self.cosmo_model = cosmo_model
        if self.cosmo_model == "FlatwCDM":
            self.cosmology_names = ["H0", "Om0", "w0"]
        elif self.cosmo_model == "FlatLambdaCDM":
            self.cosmology_names = ["H0", "Om0"]
        else:
            self.cosmology_names = []
        self._cosmo = available[cosmo_model]

    def cosmology_variables(self, parameters):
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
        return {key: parameters[key] for key in self.cosmology_names}

    def cosmology(self, parameters):
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
        if isinstance(self._cosmo, WCosmoMixin):
            return self._cosmo
        else:
            return self._cosmo(**self.cosmology_variables(parameters))

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
            cosmo = self.cosmology(self.parameters)
            samples["redshift"] = z_at_value(
                cosmo.luminosity_distance,
                data["luminosity_distance"],
            )
            jacobian = cosmo.dDLdz(samples["redshift"])
        elif "redshift" not in data:
            raise ValueError(
                f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
            )
        else:
            jacobian = xp.ones(data["redshift"].shape)

        for key in list(data.keys()):
            if key.endswith("_detector"):
                samples[key[:-9]] = data[key] / (1 + samples["redshift"])
                jacobian *= 1 + samples["redshift"]
            elif key != "luminosity_distance":
                samples[key] = data[key]

        return samples, jacobian


class CosmoModel(NonCachingModel, CosmoMixin):
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

    def __init__(self, model_functions=None, cosmo_model="Planck15"):
        NonCachingModel.__init__(self, model_functions=model_functions)
        CosmoMixin.__init__(self, cosmo_model=cosmo_model)

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

        data, jacobian = self.detector_frame_to_source_frame(data)
        probability = super().prob(data, **kwargs)
        probability /= jacobian

        return probability


class MultiCosmoMixin:
    """
    Mixin class that provides cosmological functionality to a subclass.

    Parameters
    ==========
    cosmo_model: str
        The cosmology model to use. Default is :code:`Planck15`.
        Should be of :code:`wcosmo.available.keys()`.
    """

    def __init__(self, cosmo_model_lists=None):
        wcosmo_disable_units()
        self.cosmo_model_lists = cosmo_model_lists
        self._cosmo_lists = []
        self.cosmology_name_lists = []
        for i, cosmo_model in enumerate(self.cosmo_model_lists):
            if cosmo_model == "FlatwCDM":
                self.cosmology_name_lists.append([f"H0_{i}", f"Om0_{i}", f"w0_{i}"])
            elif cosmo_model == "FlatLambdaCDM":
                self.cosmology_name_lists.append([f"H0_{i}", f"Om0_{i}"])
            else:
                self.cosmology_name_lists.append([])
            self._cosmo_lists.append(available[cosmo_model])

    def cosmology(self, parameters):
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
        cosmology_lists=list()
        for i, _cosmo in enumerate(self._cosmo_lists):
            _mapping={f'H0_{i}':'H0', f'Om0_{i}':'Om0', f'w0_{i}':'w0'}
            cosmology_lists.append(_cosmo(**{_mapping[key]: parameters[key] for key in self.cosmology_names[i]}))
        return cosmology_lists

    def detector_frame_to_source_frame(self, data, cosmo, **parameters):
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
            samples["redshift"] = z_at_value(
                cosmo.luminosity_distance,
                data["luminosity_distance"],
            )
            jacobian = cosmo.dDLdz(samples["redshift"])
        elif "redshift" not in data:
            raise ValueError(
                f"Either luminosity distance or redshift provided in detector frame to source frame samples conversion"
            )
        else:
            jacobian = xp.ones(data["redshift"].shape)

        for key in list(data.keys()):
            if key.endswith("_detector"):
                samples[key[:-9]] = data[key] / (1 + samples["redshift"])
                jacobian *= 1 + samples["redshift"]
            elif key != "luminosity_distance":
                samples[key] = data[key]

        return samples, jacobian

class MultiNonCachingModel:
    """
    Modified version of bilby.hyper.model.Model that disables caching for jax.
    """

    def __init__(self, model_function_lists=None):
        """
        Parameters
        ==========
        model_functions: list
            List of all sub populations.
        cache: bool
            Whether to cache the value returned by the model functions,
            default=:code:`True`. The caching only looks at the parameters
            not the data, so should be used with caution. The caching also
            breaks :code:`jax` JIT compilation.
        """
        self.model_lists = model_function_lists
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

    def prob(self, data, **kwargs):
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
        probability = xp.ones(len(self.model_lists))
        for i, models in enumerate(self.model_lists):
            for function in models:
                new_probability = function(data, **self._get_function_parameters(function))
                probability[i] *= new_probability
        return probability

class MultiCosmoModel(MultiNonCachingModel, MultiCosmoMixin):
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

    def __init__(self, model_function_lists=None, cosmo_model_lists=None):
        MultiNonCachingModel.__init__(self, model_function_lists=model_function_lists)
        MultiCosmoMixin.__init__(self, cosmo_model_lists=cosmo_model_lists)

    def prob(self, data_detector, **kwargs):
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
        weighted_probability = 0.
        cosmo = self.cosmology(self.parameters)
        for i, models in enumerate(self.model_lists):
            data_source, jacobian = self.detector_frame_to_source_frame(data_detector, cosmo[i])
            for function in models:
                new_probability = function(data_source, **self._get_function_parameters(function))
                probability *= new_probability
            probability /= jacobian
            weighted_probability += self.parameters['likelihood_mix{i}']*probability

        return weighted_probability

from rf_pool.models import Model

class RCNN_Model(Model):
    """
    Faster-RCNN Object-Detection Model (i.e., from `torchvision.models.detection`)

    Parameters
    ----------
    model : dict or nn.Module
        model to be set or built (if `isinstance(model, dict)`)
    **kwargs : **dict
        (method, kwargs) pairs for calling additional methods (see Methods)

    Methods
    -------
    replace_modules(**layers : **dict) : replace modules in model
    insert_modules(**layers : **dict) : insert modules into model
    set_parameters(**params : **dict) : set parameters for training
    print_model(verbose : bool) : print model and other attributes

    See Also
    --------
    rf_pool.models.Model
    rf_pool.solver.build.build_model

    Notes
    -----
    The Faster-RCNN model from `torchvision.models.detection` takes a list of
    tensors (inputs) and list of dictionaries (targets) during training and returns
    a dictionary of losses. As such, this class works with the `Solver` to pass
    both `inputs` and `targets` to the model and should be used with the `SumLoss`
    criterion for training.
    """
    def __init__(self, model, **kwargs):
        super(RCNN_Model, self).__init__(model, **kwargs)

    def forward(self, inputs, targets=None, **kwargs):
        """
        Pass inputs through model and return outputs and targets

        Parameters
        ----------
        inputs : list[torch.Tensor]
            inputs passed to RCNN model
        targets : list[dict] or None
            targets passed to RCNN model (during training, see e.g.
            `torchvision.models.detection.fasterrcnn_resnet50_fpn`)
            [default: None]
        **kwargs : **dict
            keyword arguments passed to model with `inputs`

        Returns
        -------
        outputs : any
            model outputs passed to loss function
        targets : list[dict] or None
            same as input parameter `targets` (see Notes)

        Notes
        -----
        The Faster-RCNN model from `torchvision.models.detection` takes both
        inputs and targets during training and returns a dictionary of losses.
        In order to train using the `Solver` class, the loss criterion should
        be `SumLoss`, which returns the sum of the dictionary of losses returned
        from the Faster-RCNN model and ignores the `targets` variable. If
        `self.training is False`, the `targets` are not passed to the model.
        """
        # return model outputs and targets
        if self._model.training:
            return self._model(inputs, targets, **kwargs), targets
        return self._model(inputs, **kwargs), targets

if __name__ == '__main__':
    import doctest
    doctest.testmod()

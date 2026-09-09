def get_metric_alpha(model):
    """
    Return the learned FlatVI alpha.

    Parameters
    ----------
    model
        Trained FlatVAE.

    Returns
    -------
    float
        Learned alpha.
    """
    module = model.module
    try:
        return float(module.metric_alpha.detach().cpu())
    except:
        return 0
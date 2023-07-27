import src.modules.models.mf as mf
import src.modules.models.LBD as LBD
import src.modules.models.baselines.CMF as CMF
import src.modules.models.baselines.ordrec as ordrec

import src.modules.utils.utils as utils

def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def init_compile_model(model_params, compilation_params):
    model_type = model_params["model"]

    _model_params = utils.preprocess_params(**model_params)
    _compilation_params = utils.preprocess_params(**compilation_params)
    print(_compilation_params)
    if model_type == "LBD":
        model = LBD.init_model(**_model_params)
    elif model_type == "mf":
        model = mf.init_model(**_model_params)
    elif model_type == "CMF":
        model = CMF.init_model(**_model_params)
    elif model_type == "ordrec":
        model = ordrec.init_model(**_model_params)
    else:
        raise NotImplementedError("Only supporting LBD and mf models.")
    compiled_model = compile_model(model, **_compilation_params)
    return compiled_model
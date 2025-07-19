from src.model import RBM

def get_model(model_type, **kwargs):
    if model_type.lower() == "rbm":
        return RBM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

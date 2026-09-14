import torch


def compute_parameter_epsilons(parameters, rho_w):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        return None
    norm = torch.norm(torch.stack([torch.norm(gradient, p=2) for gradient in gradients]), p=2)
    if not torch.isfinite(norm) or float(norm.item()) == 0.0:
        return None
    scale = float(rho_w) / (norm + 1e-12)
    return {
        parameter: parameter.grad.detach().clone() * scale
        for parameter in parameters if parameter.grad is not None
    }


def compute_feature_epsilon(grad, rho_z):
    if grad is None or not torch.isfinite(grad).all():
        return None
    norm = torch.norm(grad, p=2)
    if not torch.isfinite(norm) or float(norm.item()) == 0.0:
        return None
    return grad.detach().clone() * (float(rho_z) / (norm + 1e-12))


class DPOParameterPerturbation:
    """Owns temporary parameter values and restores them before SGD."""

    def __init__(self, optimizer, epsilons):
        self.optimizer = optimizer
        self.epsilons = epsilons
        self.old_parameters = {}

    def apply(self):
        with torch.no_grad():
            for parameter, epsilon in self.epsilons.items():
                self.old_parameters[parameter] = parameter.detach().clone()
                parameter.add_(epsilon)

    def restore(self):
        with torch.no_grad():
            for parameter, old_parameter in self.old_parameters.items():
                parameter.copy_(old_parameter)
        self.old_parameters.clear()

    def step(self):
        self.restore()
        self.optimizer.step()

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        return False


class DPOFusedBEVPerturbation:
    """Captures or perturbs only the post-ConvFuser fused BEV tensor."""

    def __init__(self, fuser, epsilon_z=None, capture=False):
        self.fuser = fuser
        self.epsilon_z = None if epsilon_z is None else epsilon_z.detach()
        self.capture = bool(capture)
        self.fused_bev = None
        self.handle = None

    def _hook(self, module, inputs, batch_dict):
        fused_bev = batch_dict['spatial_features']
        if self.capture:
            fused_bev.retain_grad()
            self.fused_bev = fused_bev
        if self.epsilon_z is not None:
            batch_dict['spatial_features'] = batch_dict['spatial_features'] + self.epsilon_z
        return batch_dict

    def __enter__(self):
        self.handle = self.fuser.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False

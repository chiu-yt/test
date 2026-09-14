import torch


class InvalidSAMConfigurationError(ValueError):
    """Raised when SAM is constructed with unsupported settings."""


class SAMTransactionError(RuntimeError):
    """Raised when the perturbation transaction is used out of order."""


class SAM(torch.optim.Optimizer):
    """SGD-backed Sharpness-Aware Minimization with explicit rollback."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise InvalidSAMConfigurationError('SAM rho must be non-negative')
        if base_optimizer is not torch.optim.SGD:
            raise InvalidSAMConfigurationError(
                'SAR requires torch.optim.SGD as the base optimizer'
            )

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = torch.optim.SGD(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults.update(self.base_optimizer.defaults)

    @property
    def transaction_active(self):
        """Return whether a perturbation is waiting to be restored."""
        return any('old_p' in state for state in self.state.values())

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb parameters and return gradient and perturbation diagnostics."""
        if self.transaction_active:
            raise SAMTransactionError('SAM perturbation transaction is already active')

        grad_norm = self._grad_norm()
        perturb_sq_norm = 0.0
        perturbed_params = 0
        perturbed_elements = 0
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                self.state[parameter]['old_p'] = parameter.detach().clone()
                adaptive_scale = parameter.square() if group['adaptive'] else 1.0
                perturbation = adaptive_scale * parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)
                perturb_sq_norm += float(perturbation.norm(p=2).square().item())
                perturbed_params += 1
                perturbed_elements += parameter.numel()

        if zero_grad:
            self.zero_grad()
        return {
            'grad_norm': float(grad_norm.item()),
            'perturb_norm': perturb_sq_norm ** 0.5,
            'perturbed_params': perturbed_params,
            'perturbed_elements': perturbed_elements,
        }

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore every perturbed parameter before applying the SGD update."""
        restored_params = self._restore_perturbed()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
        return {'restored_params': restored_params, 'stepped': True}

    @torch.no_grad()
    def rollback(self, zero_grad=False):
        """Restore every perturbed parameter without applying an SGD update."""
        restored_params = self._restore_perturbed()
        if zero_grad:
            self.zero_grad()
        return {'restored_params': restored_params, 'stepped': False}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise SAMTransactionError('SAM requires a closure')
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        try:
            closure()
            self.second_step()
        finally:
            if self.transaction_active:
                self.rollback(zero_grad=True)
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.state = self.state

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        grad_norms = [
            ((parameter.abs() if group['adaptive'] else 1.0) * parameter.grad)
            .norm(p=2).to(shared_device)
            for group in self.param_groups
            for parameter in group['params']
            if parameter.grad is not None
        ]
        if not grad_norms:
            return torch.zeros((), device=shared_device)
        return torch.stack(grad_norms).norm(p=2)

    def _restore_perturbed(self):
        restored_params = 0
        for group in self.param_groups:
            for parameter in group['params']:
                old_parameter = self.state[parameter].pop('old_p', None)
                if old_parameter is None:
                    continue
                parameter.copy_(old_parameter)
                restored_params += 1
        return restored_params

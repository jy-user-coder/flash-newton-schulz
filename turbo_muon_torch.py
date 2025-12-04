import torch
import torch.distributed as dist

try:
    from kernels import get_kernel
    kern = get_kernel("tboissin/newton_schulz_triton")
    newton_schulz = torch.compile(kern.newton_schulz) # optionally compile with torch.compile
except ImportError as e:
    print("Warning: could not import newton_schulz_triton kernels, falling back to PyTorch implementation.")

    def newton_schulz(G, iter=4, precondition=True, epsilon: float = 1e-7, dtype=torch.bfloat16):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert G.ndim >= 2
        ns_consts = [
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        ][-iter:]
        X = G.to(dtype=dtype)
        if G.size(-2) > G.size(-1):
            X = X.mT

        if not precondition:
            X /= (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

        # Perform the NS iterations
        for i, (a, b, c) in enumerate(ns_consts):
            A = X @ X.mT
            if precondition and i == 0:
                s = torch.rsqrt(
                    torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=epsilon)
                )
                X = X * s.unsqueeze(-1)
                A = A * s.unsqueeze(-1) * s.unsqueeze(-2)

            B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X


def muon_update(grad, momentum, precondition, beta=0.95, ns_steps=4, nesterov=True, epsilon=1e-6):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = newton_schulz(update, precondition=precondition, iter=ns_steps, epsilon=epsilon)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class TurboMuon(torch.optim.Optimizer):
    """
    TurboMuon - Turbocharing MomentUm Orthogonalized by Newton-schulz via AOL preconditioning

    Built upon:
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        variant: Either 'standard' for regular Muon usage or 'turbo' for 
        accelerated Newton-Schulz iterations using AOL preconditioning (
        c.f. https://hal.science/hal-05390446).
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, variant="turbo", lr=0.02, weight_decay=0, momentum=0.95):
        assert variant in ["standard", "turbo"]
        self.variant = variant
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        steps = {
            "standard":5,
            "turbo":4,
        }[self.variant]
        precondition = {
            "standard":False,
            "turbo":True,
        }[self.variant]

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], precondition, ns_steps=steps, beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceTurboMuon(torch.optim.Optimizer):
    """
    TurboMuon variant for usage in non-distributed settings.
    """
    def __init__(self, params, variant="turbo", lr=0.02, weight_decay=0, momentum=0.95):
        assert variant in ["standard", "turbo"]
        self.variant = variant
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        steps = {
            "standard":5,
            "turbo":4,
        }[self.variant]
        precondition = {
            "standard":False,
            "turbo":True,
        }[self.variant]

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], precondition, ns_steps=steps, beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class TurboMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed TurboMuon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with TurboMuon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["variant"] = group.get("variant", "turbo")
                assert group["variant"] in ["standard", "turbo"]
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "variant"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                steps = {
                    "standard":5,
                    "turbo":4,
                }[group["variant"]]
                precondition = {
                    "standard":False,
                    "turbo":True,
                }[group["variant"]]

                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], precondition, ns_steps=steps, beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceTurboMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of TurboMuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["variant"] = group.get("variant", "turbo")
                assert group["variant"] in ["standard", "turbo"]
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon", "variant"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                steps = {
                    "standard":5,
                    "turbo":4,
                }[group["variant"]]
                precondition = {
                    "standard":False,
                    "turbo":True,
                }[group["variant"]]
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], precondition, ns_steps=steps, beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

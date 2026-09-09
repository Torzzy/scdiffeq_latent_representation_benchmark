import math

import torch
import torch.nn as nn


def _check_finite(
    name,
    x,
    *,
    t=None,
    z_reference=None,
):
    """Check whether a tensor contains finite values.

    Parameters
    ----------
    name
        Name used to identify the tensor in diagnostic messages.
    x
        Tensor to check.
    t
        Optional time value associated with the tensor.
    z_reference
        Optional input state used for comparison.

    Returns
    -------
    bool
        ``True`` if all values are finite, otherwise ``False``.
    """
    if torch.isfinite(
        x
    ).all():
        return True

    bad = ~torch.isfinite(
        x
    )

    n_bad = bad.sum().item()
    n_total = x.numel()

    nan_count = torch.isnan(
        x
    ).sum().item()

    posinf_count = torch.isposinf(
        x
    ).sum().item()

    neginf_count = torch.isneginf(
        x
    ).sum().item()

    finite_values = x[
        torch.isfinite(x)
    ]

    print()
    print(
        "[Euler-Maruyama] "
        f"NON-FINITE VALUES in {name}"
    )
    print(
        f"shape      : {tuple(x.shape)}"
    )
    print(
        f"bad values : {n_bad}/{n_total}"
    )
    print(
        f"NaN        : {nan_count}"
    )
    print(
        f"+Inf       : {posinf_count}"
    )
    print(
        f"-Inf       : {neginf_count}"
    )

    if t is not None:
        print(
            f"time       : {t}"
        )

    if finite_values.numel() > 0:
        finite_abs = finite_values.abs()

        print(
            f"finite min : "
            f"{finite_values.min().item():.6e}"
        )
        print(
            f"finite max : "
            f"{finite_values.max().item():.6e}"
        )
        print(
            f"finite max |x| : "
            f"{finite_abs.max().item():.6e}"
        )

    if bad.ndim > 1:
        bad_rows = bad.any(
            dim=tuple(
                range(
                    1,
                    bad.ndim,
                )
            )
        )

        bad_row_indices = (
            bad_rows.nonzero(
                as_tuple=False,
            )
            .flatten()
        )

        print(
            "bad rows   : "
            f"{len(bad_row_indices)}"
        )

        print(
            "first bad rows:",
            bad_row_indices[
                :10
            ]
            .detach()
            .cpu()
            .tolist(),
        )

    indices = bad.nonzero(
        as_tuple=False,
    )

    print(
        "first bad indices:",
        indices[
            :10
        ]
        .detach()
        .cpu()
        .tolist(),
    )

    if z_reference is not None:
        z_finite = torch.isfinite(
            z_reference
        )

        print(
            "input z finite : "
            f"{z_finite.all().item()}"
        )

        if z_finite.all():
            z_abs = z_reference.abs()

            print(
                f"input z max |x| : "
                f"{z_abs.max().item():.6e}"
            )
            print(
                f"input z mean |x|: "
                f"{z_abs.mean().item():.6e}"
            )
        else:
            print(
                "WARNING: input z already "
                "contains non-finite values."
            )

    print()

    return False


def _tensor_max_abs(x):
    """Return the maximum absolute finite value of a tensor.

    Parameters
    ----------
    x
        Tensor to inspect.

    Returns
    -------
    float or None
        Maximum absolute value, ``inf`` if no finite values exist,
        or ``None`` for non-tensors and empty tensors.
    """
    if not torch.is_tensor(
        x
    ):
        return None

    if x.numel() == 0:
        return None

    finite = torch.isfinite(
        x
    )

    if not finite.any():
        return float("inf")

    return (
        x[finite]
        .abs()
        .max()
        .item()
    )


def _diagnose_module_parameters(
    module,
    *,
    threshold=1e3,
):
    """Inspect model parameters after a numerical failure.

    Parameters
    ----------
    module
        Module whose parameters are inspected.
    threshold
        Maximum expected absolute parameter value.

    Returns
    -------
    bool
        ``True`` if all parameters are finite and below the threshold.
    """
    print()
    print(
        "[Euler-Maruyama] "
        "DIAGNOSTIC: model parameters"
    )

    problem = False

    for name, parameter in module.named_parameters():
        data = parameter.detach()

        finite = torch.isfinite(
            data
        ).all().item()

        max_abs = _tensor_max_abs(
            data
        )

        if (
            not finite
            or (
                max_abs is not None
                and max_abs > threshold
            )
        ):
            problem = True

            print(
                f"[PARAMETER PROBLEM] {name}"
            )
            print(
                f"    shape    : "
                f"{tuple(data.shape)}"
            )
            print(
                f"    finite   : {finite}"
            )
            print(
                f"    max |x|  : {max_abs}"
            )

    if not problem:
        print(
            "All parameters are finite "
            f"and max |parameter| <= {threshold:.1e}"
        )

    print()

    return not problem


def _diagnose_forward_layers(
    module,
    z,
    t,
    *,
    threshold=1e3,
):
    """Inspect a diffusion network layer by layer after a failure.

    Parameters
    ----------
    module
        Diffusion module containing a ``net`` attribute.
    z
        Latent states used as network input.
    t
        Time values used as network input.
    threshold
        Activation magnitude above which a layer is considered unstable.
    """
    print()
    print(
        "[Euler-Maruyama] DIAGNOSTIC: "
        "diffusion network forward pass"
    )

    if not hasattr(
        module,
        "net",
    ):
        print(
            "Cannot inspect diffusion network: "
            "module has no `.net` attribute."
        )
        print()
        return

    net = module.net

    if not torch.is_tensor(
        t
    ):
        t = torch.tensor(
            t,
            device=z.device,
            dtype=z.dtype,
        )

    if t.ndim == 0:
        t = t.expand(
            len(z)
        )

    t = t[:, None]

    x = torch.cat(
        [
            z,
            t,
        ],
        dim=1,
    )

    max_abs = _tensor_max_abs(
        x
    )

    print(
        "[INPUT] z + t"
    )
    print(
        f"    shape   : {tuple(x.shape)}"
    )
    print(
        f"    max |x| : {max_abs:.6e}"
    )

    if not torch.isfinite(
        x
    ).all():
        print(
            "    !!! input already non-finite"
        )
        print()
        return

    current = x
    first_problem = None

    for index, layer in enumerate(
        net
    ):
        layer_name = (
            f"{index}:{layer.__class__.__name__}"
        )

        try:
            current = layer(
                current
            )
        except Exception as exc:
            print()
            print(
                f"[FORWARD ERROR] {layer_name}"
            )
            print(
                f"    error: {repr(exc)}"
            )

            first_problem = layer_name
            break

        finite = torch.isfinite(
            current
        ).all().item()

        max_abs = _tensor_max_abs(
            current
        )

        print(
            f"[LAYER] {layer_name}"
        )
        print(
            f"    shape   : "
            f"{tuple(current.shape)}"
        )
        print(
            f"    finite  : {finite}"
        )
        print(
            f"    max |x| : {max_abs:.6e}"
            if max_abs is not None
            else
            "    max |x| : None"
        )

        if (
            not finite
            or (
                max_abs is not None
                and max_abs > threshold
            )
        ):
            if first_problem is None:
                first_problem = layer_name

                print()
                print(
                    "    >>> FIRST "
                    "NUMERICAL EXPLOSION <<<"
                )

    print()

    if first_problem is not None:
        print(
            "FIRST PROBLEMATIC LAYER:"
        )
        print(
            f"    {first_problem}"
        )
    else:
        print(
            "No layer exceeded the diagnostic "
            f"threshold ({threshold:.1e})."
        )

    print()


def _diagnose_diffusion(
    diffusion_module,
    z,
    t,
):
    """Run a full diagnostic after a diffusion failure.

    Parameters
    ----------
    diffusion_module
        Diffusion module that produced the failure.
    z
        Latent state at the time of failure.
    t
        Time value at the time of failure.
    """
    print()
    print(
        "[Euler-Maruyama] "
        "DIFFUSION FAILURE DIAGNOSTIC"
    )

    z_finite = torch.isfinite(
        z
    ).all().item()

    z_max = _tensor_max_abs(
        z
    )

    print()
    print(
        "[1] INPUT STATE z"
    )
    print(
        f"    finite  : {z_finite}"
    )
    print(
        f"    max |z| : {z_max:.6e}"
        if z_max is not None
        else
        "    max |z| : None"
    )

    _diagnose_module_parameters(
        diffusion_module,
    )

    if z_finite:
        _diagnose_forward_layers(
            diffusion_module,
            z,
            t,
        )

    print()


class EulerMaruyamaSimulator:
    """Euler-Maruyama simulator for latent stochastic differential equations.

    The simulated SDE is

    ``dX = f(X, t) dt + g(X, t) dW``.
    """

    def __init__(
        self,
        sde,
        dt=0.05,
    ):
        """Initialize the Euler-Maruyama simulator.

        Parameters
        ----------
        sde
            SDE model providing drift and diffusion functions.
        dt
            Simulation time step.
        """
        self.sde = sde
        self.dt = dt

    def step(
        self,
        z,
        t,
    ):
        """Perform one Euler-Maruyama integration step.

        Parameters
        ----------
        z
            Current latent state.
        t
            Current simulation time.

        Returns
        -------
        torch.Tensor
            Latent state after one simulation step.

        Raises
        ------
        FloatingPointError
            If any intermediate or final value is non-finite.
        """
        if not _check_finite(
            "input z",
            z,
            t=t,
        ):
            raise FloatingPointError(
                "Euler-Maruyama received "
                "non-finite z."
            )

        drift = self.sde.f(
            t,
            z,
        )

        if not _check_finite(
            "drift f(t,z)",
            drift,
            t=t,
            z_reference=z,
        ):
            raise FloatingPointError(
                "SDE drift produced "
                "non-finite values."
            )

        try:
            sigma = self.sde.g(
                t,
                z,
            )
        except Exception:
            _diagnose_diffusion(
                self.sde.g,
                z,
                t,
            )
            raise

        if not _check_finite(
            "diffusion g(t,z)",
            sigma,
            t=t,
            z_reference=z,
        ):
            _diagnose_diffusion(
                self.sde.g,
                z,
                t,
            )

            raise FloatingPointError(
                "SDE diffusion produced "
                "non-finite values."
            )

        eps = torch.randn_like(
            z
        )

        if not _check_finite(
            "random noise eps",
            eps,
            t=t,
        ):
            raise FloatingPointError(
                "Random noise produced "
                "non-finite values."
            )

        drift_term = (
            drift
            * self.dt
        )

        if not _check_finite(
            "drift * dt",
            drift_term,
            t=t,
        ):
            raise FloatingPointError(
                "Drift Euler term "
                "became non-finite."
            )

        noise_scale = math.sqrt(
            self.dt
        )

        diffusion_term = (
            sigma
            * noise_scale
            * eps
        )

        if not _check_finite(
            "sigma * sqrt(dt) * eps",
            diffusion_term,
            t=t,
        ):
            _diagnose_diffusion(
                self.sde.g,
                z,
                t,
            )

            raise FloatingPointError(
                "Diffusion Euler term "
                "became non-finite."
            )

        z_next = (
            z
            + drift_term
            + diffusion_term
        )

        if not _check_finite(
            "z_next",
            z_next,
            t=t,
        ):
            raise FloatingPointError(
                "Euler-Maruyama update "
                "produced non-finite z_next."
            )

        return z_next

    def simulate(
        self,
        z0,
        t0,
        t1,
    ):
        """Simulate an SDE trajectory between two time points.

        Parameters
        ----------
        z0
            Initial latent state.
        t0
            Initial simulation time.
        t1
            Final simulation time.

        Returns
        -------
        torch.Tensor
            Final latent state.
        """
        z = z0
        t = t0

        n_steps = int(
            round(
                (t1 - t0)
                / self.dt
            )
        )

        for step_idx in range(
            n_steps
        ):
            try:
                z = self.step(
                    z,
                    t,
                )
            except FloatingPointError:
                print(
                    "[Euler-Maruyama] "
                    f"failure at step "
                    f"{step_idx}/{n_steps}, "
                    f"t={t}"
                )
                raise

            t += self.dt

        return z

    def sample_population(
        self,
        z0,
        t0,
        t1,
        n_samples,
    ):
        """Simulate multiple trajectories from each initial state.

        Parameters
        ----------
        z0
            Initial latent states with shape ``(B, D)``.
        t0
            Initial simulation time.
        t1
            Final simulation time.
        n_samples
            Number of trajectories sampled per initial state.

        Returns
        -------
        torch.Tensor
            Simulated population with shape ``(B, n_samples, D)``.
        """
        B, d = z0.shape

        z = (
            z0[:, None]
            .expand(
                B,
                n_samples,
                d,
            )
            .reshape(
                B * n_samples,
                d,
            )
        )

        z = self.simulate(
            z,
            t0,
            t1,
        )

        return z.reshape(
            B,
            n_samples,
            d,
        )

    def continue_simulation(
        self,
        z,
        t0,
        t1,
    ):
        """Continue simulation from an existing population.

        Parameters
        ----------
        z
            Population states with shape ``(B, n_samples, D)``.
        t0
            Initial simulation time.
        t1
            Final simulation time.

        Returns
        -------
        torch.Tensor
            Continued population with shape ``(B, n_samples, D)``.
        """
        B, n_samples, d = z.shape

        z = z.reshape(
            B * n_samples,
            d,
        )

        z = self.simulate(
            z,
            t0,
            t1,
        )

        return z.reshape(
            B,
            n_samples,
            d,
        )

    def simulate_trajectory(
        self,
        z0,
        t0,
        t1,
    ):
        """Simulate a trajectory while computing drift and diffusion statistics.

        Parameters
        ----------
        z0
            Initial latent states.
        t0
            Initial simulation time.
        t1
            Final simulation time.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Final latent state and trajectory statistics.
        """
        z = z0
        t = t0

        n_steps = int(
            round(
                (t1 - t0)
                / self.dt
            )
        )

        trajectory_length = 0.0

        drift_sum = 0.0
        drift_max = 0.0

        diffusion_sum = 0.0
        diffusion_max = 0.0

        ratio_sum = 0.0
        ratio_sq_sum = 0.0
        ratio_max = 0.0

        eps = 1e-8

        for step_idx in range(
            n_steps
        ):
            if not _check_finite(
                "trajectory z",
                z,
                t=t,
            ):
                raise FloatingPointError(
                    "Non-finite trajectory state."
                )

            drift = self.sde.f(
                t,
                z,
            )

            if not _check_finite(
                "trajectory drift f(t,z)",
                drift,
                t=t,
                z_reference=z,
            ):
                raise FloatingPointError(
                    "Non-finite trajectory drift."
                )

            diffusion = self.sde.g(
                t,
                z,
            )

            if not _check_finite(
                "trajectory diffusion g(t,z)",
                diffusion,
                t=t,
                z_reference=z,
            ):
                _diagnose_diffusion(
                    self.sde.g,
                    z,
                    t,
                )

                raise FloatingPointError(
                    "Non-finite trajectory diffusion."
                )

            drift_norm = torch.norm(
                drift,
                dim=1,
            )

            diffusion_norm = torch.norm(
                diffusion,
                dim=1,
            )

            ratio = (
                drift_norm
                / (
                    drift_norm
                    + diffusion_norm
                    + eps
                )
            )

            drift_sum += (
                drift_norm.mean()
            )

            diffusion_sum += (
                diffusion_norm.mean()
            )

            drift_max = max(
                drift_max,
                drift_norm.max().item(),
            )

            diffusion_max = max(
                diffusion_max,
                diffusion_norm.max().item(),
            )

            ratio_sum += (
                ratio.mean()
            )

            ratio_sq_sum += (
                ratio ** 2
            ).mean()

            ratio_max = max(
                ratio_max,
                ratio.max().item(),
            )

            z_next = self.step(
                z,
                t,
            )

            trajectory_length += (
                torch.norm(
                    z_next - z,
                    dim=1,
                ).mean()
            )

            z = z_next

            t += self.dt

        n = float(
            n_steps
        )

        ratio_mean = (
            ratio_sum / n
        )

        ratio_variance = (
            ratio_sq_sum / n
            - ratio_mean ** 2
        )

        ratio_variance = torch.clamp(
            ratio_variance,
            min=0.0,
        )

        ratio_std = torch.sqrt(
            ratio_variance
        )

        metrics = {
            "trajectory_length":
                trajectory_length.item()
                / n,
            "ratio_mean":
                ratio_mean.item(),
            "ratio_std":
                ratio_std.item(),
            "ratio_max":
                ratio_max,
            "drift_mean":
                (drift_sum / n).item(),
            "drift_max":
                drift_max,
            "diffusion_mean":
                (diffusion_sum / n).item(),
            "diffusion_max":
                diffusion_max,
        }

        return z, metrics

    def sample_population_trajectory(
        self,
        z0,
        t0,
        t1,
        n_samples,
    ):
        """Simulate population trajectories from initial states.

        Parameters
        ----------
        z0
            Initial latent states with shape ``(B, D)``.
        t0
            Initial simulation time.
        t1
            Final simulation time.
        n_samples
            Number of trajectories sampled per initial state.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Simulated population trajectories and trajectory statistics.
        """
        B, d = z0.shape

        z = (
            z0[:, None]
            .expand(
                B,
                n_samples,
                d,
            )
            .reshape(
                B * n_samples,
                d,
            )
        )

        z, metrics = (
            self.simulate_trajectory(
                z,
                t0,
                t1,
            )
        )

        z = z.reshape(
            B,
            n_samples,
            d,
        )

        return z, metrics

    def continue_trajectory(
        self,
        z,
        t0,
        t1,
    ):
        """Continue population trajectories while collecting statistics.

        Parameters
        ----------
        z
            Population states with shape ``(B, n_samples, D)``.
        t0
            Initial simulation time.
        t1
            Final simulation time.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Continued population trajectories and trajectory statistics.
        """
        B, n_samples, d = z.shape

        z = z.reshape(
            B * n_samples,
            d,
        )

        z, metrics = (
            self.simulate_trajectory(
                z,
                t0,
                t1,
            )
        )

        z = z.reshape(
            B,
            n_samples,
            d,
        )

        return z, metrics
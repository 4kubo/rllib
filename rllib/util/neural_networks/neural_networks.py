"""Implementation of different Neural Networks with pytorch."""

import torch
import torch.jit
import torch.nn as nn

from rllib.util.utilities import safe_cholesky
from .utilities import inverse_softplus, parse_layers, update_parameters


class FeedForwardNN(nn.Module):
    """Feed-Forward Neural Network Implementation.

    Parameters
    ----------
    in_dim: Tuple[int]
        input dimension of neural network.
    out_dim: Tuple[int]
        output dimension of neural network.
    layers: list of int, optional
        list of width of neural network layers, each separated with a non-linearity.
    biased_head: bool, optional
        flag that indicates if head of NN has a bias term or not.

    """

    def __init__(
            self,
            in_dim,
            out_dim,
            layers=(),
            non_linearity="Tanh",
            biased_head=True,
            squashed_output=False,
            initial_scale=0.5,
            log_scale=False,
            min_scale=None,
            max_scale=None,
    ):
        super().__init__()
        self.kwargs = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "layers": layers,
            "non_linearity": non_linearity,
            "biased_head": biased_head,
            "squashed_output": squashed_output,
            "initial_scale": initial_scale,
            "log_scale": log_scale,
        }

        self.hidden_layers, in_dim = parse_layers(layers, in_dim, non_linearity)
        self.embedding_dim = in_dim + 1 if biased_head else in_dim
        self.head = nn.Linear(in_dim, out_dim[0], bias=biased_head)
        self.squashed_output = squashed_output
        self.log_scale = log_scale
        if self.log_scale:
            init_scale_transformed = torch.log(torch.tensor([initial_scale]))
            self._min_scale = min_scale or -20.0
            self._max_scale = max_scale or 2.0
        else:
            init_scale_transformed = inverse_softplus(torch.tensor([initial_scale]))
            self._min_scale = min_scale or 1e-6
            self._max_scale = max_scale or 1

        self._init_scale_transformed = nn.Parameter(
            init_scale_transformed, requires_grad=False
        )

    @classmethod
    def from_other(cls, other, copy=True):
        """Initialize Feedforward NN from other NN Network."""
        out = cls(**other.kwargs)
        if copy:
            update_parameters(target_module=out, new_module=other)
        return out

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x out_dim].
        """
        x = x.unsqueeze(0)  # Always un-squeeze to produce a batch.
        out = self.head(self.hidden_layers(x))
        out = out.squeeze(0)  # Squeeze back!
        if self.squashed_output:
            return torch.tanh(out)
        return out

    @torch.jit.export
    def last_layer_embeddings(self, x):
        """Get last layer embeddings of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.Tensor.
            Tensor of size [batch_size x embedding_dim].
        """
        out = self.hidden_layers(x)
        if self.head.bias is not None:
            out = torch.cat((out, torch.ones(out.shape[:-1] + (1,))), dim=-1)

        return out


class DeterministicNN(FeedForwardNN):
    """Declaration of a Deterministic Neural Network."""

    pass


class HeteroGaussianNN(FeedForwardNN):
    """A Module that parametrizes a diagonal heteroscedastic Normal distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale = nn.Linear(
            in_features=self.head.in_features,
            out_features=self.kwargs["out_dim"][0],
            bias=self.kwargs["biased_head"],
        )

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor.
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        mean = self.head(x)
        if self.squashed_output:
            mean = torch.tanh(mean)

        if self.log_scale:
            log_scale = self._scale(x).clamp(self._min_scale, self._max_scale)
            scale = torch.exp(log_scale + self._init_scale_transformed)
        else:
            scale = nn.functional.softplus(
                self._scale(x) + self._init_scale_transformed
            ).clamp(self._min_scale, self._max_scale)
        return mean, torch.diag_embed(scale)


class HomoGaussianNN(FeedForwardNN):
    """A Module that parametrizes a diagonal homoscedastic Normal distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_dim = self.kwargs["out_dim"]
        if self.log_scale:
            initial_scale = self._min_scale * torch.ones(out_dim)
        else:
            initial_scale = inverse_softplus(self._min_scale * torch.ones(out_dim))
        self._scale = nn.Parameter(initial_scale, requires_grad=True)

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        out: torch.distributions.MultivariateNormal
            Multivariate distribution with mean of size [batch_size x out_dim] and
            covariance of size [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        mean = self.head(x)
        if self.squashed_output:
            mean = torch.tanh(mean)

        if self.log_scale:
            log_scale = self._scale.clamp(self._min_scale, self._max_scale)
            scale = torch.exp(log_scale + self._init_scale_transformed)
        else:
            scale = nn.functional.softplus(
                self._scale + self._init_scale_transformed
            ).clamp(self._min_scale, self._max_scale)

        return mean, torch.diag_embed(scale)


class CategoricalNN(FeedForwardNN):
    """A Module that parametrizes a Categorical distribution."""

    pass


class Ensemble(HeteroGaussianNN):
    """Ensemble of Deterministic Neural Networks.

    TODO: Ensemble of Discrete Outputs.

    The Ensemble shares the inner layers and then has `num_heads' different heads.
    Using these heads, it returns a Multivariate Normal distribution. How these are
    computed depends on the `prediction_strategy' used.

    Parameters
    ----------
    in_dim: Tuple
        input dimension of neural network.
    out_dim: Tuple
        output dimension of neural network.
    num_heads: int
        number of heads of ensemble
    prediction_strategy: str
        - 'moment_matching': mean and covariance are computed as sample averages.
        - 'sample_head': sample a head U.A.R. and return its output.
        The same head is used along a batch.
        - 'ts_one': sample a head U.A.R. and return its output.
        A different random head is used for each element of a batch.
        - 'set_head': set a single head with .set_head() and return its output.
        This is useful for Thompson's Sampling (for example).
        - 'ts_inf': set a head with .set_head_idx() and return its output.
        Crucially, it has to have the same batch_size as the predicted state-actions.
    """

    num_heads: int
    head_ptr: int
    prediction_strategy: str
    _batch_shape: list
    _batch_rank: int
    _dim_out: int

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        prediction_strategy="moment_matching",
        layers=(),
        non_linearity="Tanh",
        biased_head=True,
        squashed_output=False,
        initial_scale=0.5,
        deterministic=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_dim,
            (out_dim[0] * num_heads,),
            layers=layers,
            non_linearity=non_linearity,
            biased_head=biased_head,
            squashed_output=squashed_output,
            initial_scale=initial_scale,
        )

        self.kwargs.update(
            out_dim=out_dim,
            num_heads=num_heads,
            prediction_strategy=prediction_strategy,
        )
        self.num_heads = num_heads
        self.head_ptr = 0
        self.head_idx = None
        self.deterministic = deterministic
        self.prediction_strategy = prediction_strategy

    @classmethod
    def from_feedforward(cls, other, num_heads, prediction_strategy="moment_matching"):
        """Initialize from a feed-forward network."""
        return cls(
            **other.kwargs,
            num_heads=num_heads,
            prediction_strategy=prediction_strategy,
            deterministic=isinstance(other, DeterministicNN),
        )

    def forward(self, x):
        """Execute forward computation of the Neural Network.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)
        out = self.head(x)

        out = torch.reshape(out, out.shape[:-1] + (-1, self.num_heads))

        if self.deterministic:
            scale = torch.zeros_like(out)
        else:
            scale_ = self._scale(x) + self._init_scale_transformed
            scale_ = self._max_scale - nn.functional.softplus(self._max_scale - scale_)
            scale_ = self._min_scale + nn.functional.softplus(scale_ - self._min_scale)
            scale = torch.reshape(scale_, scale_.shape[:-1] + (-1, self.num_heads))

        if self.prediction_strategy == "moment_matching":
            mean = out.mean(-1)
            variance = (scale.square() + out.square()).mean(-1) - mean.square()
            scale = safe_cholesky(torch.diag_embed(variance))
        elif self.prediction_strategy == "sample_head":  # Variant of TS-1
            head_ptr = torch.randint(self.num_heads, (1,))
            mean = out[..., head_ptr].squeeze(-1)
            scale = torch.diag_embed(scale[..., head_ptr].squeeze(-1))
        elif self.prediction_strategy in ["set_head", "posterior"]:  # Thompson sampling
            mean = out[..., self.head_ptr]
            scale = torch.diag_embed(scale[..., self.head_ptr])
        elif self.prediction_strategy == "ts_one":  # TS-1
            head_idx = self._sample_idx()
            mean = out.gather(-1, head_idx).squeeze(-1)
            scale = torch.diag_embed(scale.gather(-1, head_idx).squeeze(-1))
        elif self.prediction_strategy == "ts_inf":  # TS-INF
            mean = out.gather(-1, self.head_idx).squeeze(-1)
            scale = torch.diag_embed(scale.gather(-1, self.head_idx).squeeze(-1))
        else:
            raise NotImplementedError

        return mean, scale

    @torch.jit.export
    def set_head(self, new_head: int):
        """Set the Ensemble head.

        Parameters
        ----------
        new_head: int
            If new_head == num_heads, then forward returns the average of all heads.
            If new_head < num_heads, then forward returns the output of `new_head' head.

        Raises
        ------
        ValueError: If new_head > num_heads.
        """
        self.head_ptr = new_head
        if not (0 <= self.head_ptr < self.num_heads):
            raise ValueError("head_ptr has to be between zero and num_heads - 1.")

    @torch.jit.export
    def get_head(self) -> int:
        """Get current head."""
        return self.head_ptr

    @torch.jit.export
    def set_head_idx(self, head_idx):
        """Set ensemble head for particles.."""
        self.head_idx = head_idx

    @torch.jit.export
    def get_head_idx(self):
        """Get ensemble head index."""
        return self.head_idx

    @torch.jit.export
    def set_prediction_strategy(self, prediction: str, shape=None):
        """Set ensemble prediction strategy."""
        self.prediction_strategy = prediction
        if shape is not None:
            self._batch_shape = shape[:-1]
            self._batch_rank = len(shape[:-1])
            self._dim_out = shape[-1]

            if prediction == "ts_inf":
                self.head_idx = self._sample_idx()

    @torch.jit.export
    def get_prediction_strategy(self) -> str:
        """Get ensemble head."""
        return self.prediction_strategy

    def _sample_idx(self, shape=None):
        if shape is None:
            batch_shape = self._batch_shape
            batch_rank = self._batch_rank
            dim_out = self._dim_out
        else:
            batch_shape = shape[:-1]
            batch_rank = len(shape[:-1])
            dim_out = shape[-1]

        indexes = torch.randint(self.num_heads, batch_shape, device=self.device)[
            ..., None, None
        ].repeat([1] * batch_rank + [dim_out, 1])
        return indexes

    @property
    def device(self):
        if not hasattr(self, "_device"):
            self._device = next(self.parameters()).device
        return self._device


class FelixNet(FeedForwardNN):
    """A Module that implements FelixNet."""

    def __init__(self, in_dim, out_dim, initial_scale=0.5):
        super().__init__(
            in_dim,
            out_dim,
            layers=(64, 64),
            non_linearity="Tanh",
            squashed_output=True,
            biased_head=False,
            initial_scale=initial_scale,
        )
        self.kwargs = {"in_dim": in_dim, "out_dim": out_dim}

        torch.nn.init.zeros_(self.hidden_layers[0].bias)
        torch.nn.init.zeros_(self.hidden_layers[2].bias)
        self._scale = nn.Linear(64, out_dim[0], bias=False)

    def forward(self, x):
        """Execute forward computation of FelixNet.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of size [batch_size x in_dim] where the NN is evaluated.

        Returns
        -------
        mean: torch.Tensor.
            Mean of size [batch_size x out_dim].
        scale_tril: torch.Tensor.
            Cholesky factorization of covariance matrix of size.
            [batch_size x out_dim x out_dim].
        """
        x = self.hidden_layers(x)

        mean = torch.tanh(self.head(x))
        scale = nn.functional.softplus(
            self._scale(x) + self._init_scale_transformed
        ).clamp(self._min_scale, self._max_scale)
        return mean, torch.diag_embed(scale)

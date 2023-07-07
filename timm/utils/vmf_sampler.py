# coding=utf-8
# Copyright 2021 The vMF Embeddings Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""torch.distributions.Distribution implementation of a von Mises-Fisher.

Code was adapted from:
    https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py
"""

import math
import torch

EPS = 1e-14


class VonMisesFisher(torch.distributions.Distribution):
  """torch.distributions.Distribution implementation of a von Mises-Fisher."""

  arg_constraints = {
      "loc": torch.distributions.constraints.real,
      "scale": torch.distributions.constraints.positive,
  }
  support = torch.distributions.constraints.real
  has_rsample = True
  _mean_carrier_measure = 0

  def __init__(self, loc, scale, validate_args=None, k=1):
    self.dtype = loc.dtype
    self.loc = loc
    self.scale = scale
    self.device = loc.device
    self.__m = loc.shape[-1]
    self.__e1 = (torch.zeros(loc.shape[-1], device=self.device))
    self.__e1[0] = 1.0
    self.k = k
    self.log_norm_const = None  # for log_prob (if ever used)

    super().__init__(self.loc.size(), validate_args=validate_args)

  def sample(self, shape=torch.Size()):
    with torch.no_grad():
      return self.rsample(shape)

  def rsample(self, shape=torch.Size()):
    shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

    # only use __sample_w3 for 3D vMFs, otherwise __sample_w_rej
    # This samples the 1 dimensional "mixture variable" that indicates how far we are from the mode (1, 0, ..., 0)
    w = (
        self.__sample_w3(shape=shape) if self.__m == 3 else self.__sample_w_rej(
            shape=shape))

    # Draw uniform points on the unit sphere for the m-1 "other" variables
    v = (torch.distributions.Normal(
        torch.tensor(0, dtype=self.dtype, device=self.device),
        torch.tensor(1, dtype=self.dtype, device=self.device),
    ).sample(shape + torch.Size(self.loc.shape)).transpose(
        0, -1)[1:]).transpose(0, -1)
    v = v / v.norm(dim=-1, keepdim=True)

    # Build together the vector from the 1-dim rejection sample and the other m-1 dims
    w_ = torch.sqrt(torch.clamp(1 - (w**2), EPS))
    x = torch.cat((w, w_ * v), -1)

    # rotate to get the modal value from (1, 0, ..., 0) to the intended mu
    z = self.__householder_rotation(x)
    z = z.type(self.dtype)

    # One last sanity check because this sometimes returns NaN
    if torch.any(torch.isnan(z)):
        return self.rsample(shape)
    else:
        return z

  def __sample_w3(self, shape):
    shape = shape + torch.Size(self.scale.shape)
    u = (
        torch.distributions.Uniform(
            torch.tensor(0, dtype=self.dtype, device=self.device),
            torch.tensor(1, dtype=self.dtype, device=self.device),
        ).sample(shape))
    self.__w = (1 + torch.stack(
        [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0).logsumexp(0) /
                self.scale)
    return self.__w

  def __sample_w_rej(self, shape):
    c = torch.sqrt((4 * (self.scale**2)) + (self.__m - 1)**2)
    b_true = (-2 * self.scale + c) / (self.__m - 1)

    # Using Taylor approximation with a smooth swift from 10 < scale < 11 to
    # avoid numerical errors for large scale.
    b_app = (self.__m - 1) / (4 * self.scale)
    s = torch.min(
        torch.max(
            torch.tensor([0.0], dtype=self.dtype, device=self.device),
            self.scale - 10,
        ),
        torch.tensor([1.0], dtype=self.dtype, device=self.device),
    )
    b = b_app * s + b_true * (1 - s)

    a = (self.__m - 1 + 2 * self.scale + c) / 4
    d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

    self.__b, (self.__e, self.__w) = b, self.__while_loop(
        b, a, d, shape, k=self.k)
    return self.__w

  @staticmethod
  def first_nonzero(x, dim, invalid_val=-1):
    mask = x > 0
    idx = torch.where(
        mask.any(dim=dim),
        mask.float().max(dim=1)[1].squeeze(),
        torch.tensor(invalid_val, device=x.device),
    )
    return idx

  def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
    # Matrix while loop: samples a matrix of [A, k] samples, to avoid looping
    # all together.
    is_inf = self.scale == float("Inf")
    b, a, d, is_inf = [
        e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
        for e in (b, a, d, is_inf)
    ]
    w, e, bool_mask = (
        torch.zeros_like(b, device=self.device),
        torch.zeros_like(b, device=self.device),
        (torch.ones_like(b, device=self.device) == 1),
    )

    sample_shape = torch.Size([b.shape[0], k])
    shape = shape + torch.Size(self.scale.shape)

    while bool_mask.sum() != 0:
      con1 = torch.tensor(
          (self.__m - 1) / 2, dtype=torch.float64, device=self.device)
      con2 = torch.tensor(
          (self.__m - 1) / 2, dtype=torch.float64, device=self.device)
      e_ = (
          torch.distributions.Beta(con1, con2).sample(sample_shape).type(self.dtype))

      u = (
          torch.distributions.Uniform(
              torch.tensor(0 + eps, dtype=self.dtype, device=self.device),
              torch.tensor(1 - eps, dtype=self.dtype, device=self.device),
          ).sample(sample_shape).type(self.dtype))

      w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
      t = (2 * a * b) / (1 - (1 - b) * e_)

      accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)

      # For samples with infinite kappa, return the mean (by just returning a w_ = 1)
      w_[is_inf] = 1
      accept[is_inf] = True

      accept_idx = self.first_nonzero(
          accept, dim=-1, invalid_val=-1).unsqueeze(1)
      accept_idx_clamped = accept_idx.clamp(0)
      w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
      e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

      reject = accept_idx < 0
      accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

      w[bool_mask * accept] = w_[bool_mask * accept]
      e[bool_mask * accept] = e_[bool_mask * accept]

      bool_mask[bool_mask * accept] = reject[bool_mask * accept]

    return e.reshape(shape), w.reshape(shape)

  def __householder_rotation(self, x):
    u = self.__e1 - self.loc
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(EPS)
    z = x - 2 * (x * u).sum(-1, keepdim=True) * u
    return z

  def log_prob(self, value):
    # Input: value - [batch, vMF, ?,dim]
    # Output: density of value under vMF WITHOUT NORMALIZING CONSTANTS, because their derivative is not implemented

    log_p = self.scale.unsqueeze(0) * torch.sum(value * self.loc.unsqueeze(0), dim=-1, keepdim=True)
    log_p = log_p.squeeze(-1)

    return log_p

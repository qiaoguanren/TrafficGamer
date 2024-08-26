# BSD 3-Clause License

# Copyright (c) 2024, Guanren Qiao

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import deque
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Nu(nn.Module):
    """
    Class for Lagrangian multiplier.
    :param penalty_init: The value with which to initialize the Lagrange multiplier with
    :param min_clamp    : The minimum value to allow nu to drop to. If None, is set to penalty_init
    :param max_clamp    : The maximum value to allow nu to drop to. If None, is set to inf
    """

    def __init__(self, penalty_init=1., min_clamp=None, max_clamp=None):
        super(Nu, self).__init__()
        self.penalty_init = penalty_init
        penalty_init = np.log(max(np.exp(penalty_init) - 1, 1e-8))
        self.log_nu = nn.Parameter(torch.from_numpy(penalty_init) * torch.ones(1))  # for the soft-plus function
        self.min_clamp = self.penalty_init
        self.max_clamp = float('inf') if max_clamp is None else max_clamp

    def forward(self):
        # return self.log_nu.exp()
        return F.softplus(self.log_nu)

    def clamp(self):
        self.log_nu.data.clamp_(
            min=math.log(max(math.exp(self.min_clamp[0]), 1e-8)),  # min=np.log(max(np.exp(self.clamp_at)-1, 1e-8)))
            max=self.max_clamp
        )


class DualVariable:
    """
    Class for handling the Lagrangian multiplier.

    :param alpha: The budget size
    :param learning_rate: Learning rate for the Lagrange multiplier
    :param penalty_init: The value with which to initialize the Lagrange multiplier with
    """

    def __init__(self, alpha=0, learning_rate=10, penalty_init=1, min_clamp=None, max_clamp=None):
        self.nu = Nu(penalty_init, min_clamp=min_clamp, max_clamp=max_clamp)
        self.alpha = alpha
        self.loss = torch.tensor(0)
        self.optimizer = optim.Adam(
            self.nu.parameters(), lr=learning_rate[0])

    def update_parameter(self, cost):
        # Compute loss.
        self.loss = - self.nu() * (cost - self.alpha)
        # Update.
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Clamp.
        self.nu.clamp()
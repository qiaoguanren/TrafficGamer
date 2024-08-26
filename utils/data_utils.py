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

from utils.utils import add_new_agent
import torch
import math

def expand_data(data, scenario, agent_index):
    new_input_data = data
    if scenario == 4:
        v0_x = 1*math.cos(0.1)
        v0_y = 1*math.sin(0.1)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 0.1, -8379.8809, -828)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([-8341, -826]).cuda()
        v0_x = 1*math.cos(3.18)
        v0_y = 1*math.sin(3.18)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 3.18, -8311, -823)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([-8341, -826]).cuda()
        v0_x = 1*math.cos(1.6)
        v0_y = 1*math.sin(1.6)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 1.6, -8339, -863)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([-8341, -826]).cuda()
        v0_x = 1*math.cos(4.76)
        v0_y = 1*math.sin(4.76)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 4.76, -8345, -793)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([-8341, -826]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([-8341, -826]).cuda()
    elif scenario == 2:
        v0_x = 1 * math.cos(1.28)
        v0_y = 1 * math.sin(1.28)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, 1.28, 2673, -2410)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2691, -2365]).cuda()
        v0_x = 1 * math.cos(1.28)
        v0_y = 1 * math.sin(1.28)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, 1.28, 2680, -2403)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2691, -2365]).cuda()
        v0_x = 1 * math.cos(-1.95)
        v0_y = 1 * math.sin(-1.95)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, -1.95, 2693, -2340)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2681, -2376]).cuda()
        v0_x = 1 * math.cos(-1.95)
        v0_y = 1 * math.sin(-1.95)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, -1.95, 2697, -2338)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2681, -2376]).cuda()
        v0_x = 1 * math.cos(2.8)
        v0_y = 1 * math.sin(2.8)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, 2.8, 2725, -2381)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2682, -2366]).cuda()
        v0_x = 1 * math.cos(-0.35)
        v0_y = 1 * math.sin(-0.35)
        new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, -0.35, 2655, -2363)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2691, -2373]).cuda()
        # v0_x = 1 * math.cos(-0.35)
        # v0_y = 1 * math.sin(-0.35)
        # new_input_data = add_new_agent(new_input_data, 0.1, v0_x, v0_y, -0.35, 2666, -2363)
        # new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([2691, -2365]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([2691, -2365]).cuda()
    elif scenario == 3:
        v0_x = 1*math.cos(2.2)
        v0_y = 1*math.sin(2.2)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.2, 5067, 3003)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5040, 3019]).cuda()
        v0_x = 1*math.cos(-1.5)
        v0_y = 1*math.sin(-1.5)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -1.5, 5028, 3066)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5028, 3036]).cuda()
        v0_x = 1*math.cos(-1.5)
        v0_y = 1*math.sin(-1.5)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -1.6, 5032.5, 3061.5)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5033, 3043]).cuda()
        v0_x = 1*math.cos(-1.5)
        v0_y = 1*math.sin(-1.5)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -1.6, 5029, 3052)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5028, 3036]).cuda()
        v0_x = 1*math.cos(2.8)
        v0_y = 1*math.sin(2.8)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.8, 5053.5, 3017)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5040, 3019]).cuda()
        v0_x = 1*math.cos(2.7)
        v0_y = 1*math.sin(2.7)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.7, 5065, 3011.5)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5040, 3019]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([5055, 3020]).cuda()
    elif scenario == 1:
        v0_x = 1*math.cos(2.8)
        v0_y = 1*math.sin(2.8)
        new_input_data=add_new_agent(data,0.1, v0_x, v0_y, 2.8, 5208, 129)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5197,136]).cuda()
        v0_x = 1*math.cos(2.8)
        v0_y = 1*math.sin(2.8)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.8, 5208, 125)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5181,133]).cuda()
        v0_x = 1*math.cos(-1.2)
        v0_y = 1*math.sin(-1.2)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -1.2, 5183, 159)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5197,130]).cuda()
        v0_x = 1*math.cos(-1.2)
        v0_y = 1*math.sin(-1.2)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -1.2, 5185, 153)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5197,130]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([5193,145]).cuda()
    elif scenario == 5:
        v0_x = 1*math.cos(2.3)
        v0_y = 1*math.sin(2.3)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.3, 5763, 3249)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5740,3270]).cuda()
        v0_x = 1*math.cos(2.5)
        v0_y = 1*math.sin(2.5)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.5, 5755, 3251)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5740,3270]).cuda()
        v0_x = 1*math.cos(2.5)
        v0_y = 1*math.sin(2.5)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 2.5, 5759, 3254)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5740,3270]).cuda()
        v0_x = 1*math.cos(1.0)
        v0_y = 1*math.sin(1.0)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 1.0, 5720, 3243)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5728, 3261]).cuda()
        v0_x = 1*math.cos(1.0)
        v0_y = 1*math.sin(1.0)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 1.0, 5712, 3237)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5728, 3261]).cuda()
        v0_x = 1*math.cos(1.0)
        v0_y = 1*math.sin(1.0)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 1.0, 5714, 3245)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([5728, 3261]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([5740,3270]).cuda()
    elif scenario == 6:
        v0_x = 1*math.cos(0.4)
        v0_y = 1*math.sin(0.4)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 0.4, 3549, 432)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([3570, 442]).cuda()
        v0_x = 1*math.cos(0.4)
        v0_y = 1*math.sin(0.4)
        new_input_data=add_new_agent(new_input_data,0.5, v0_x, v0_y, 0.4, 3536, 426)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([3570, 442]).cuda()
        v0_x = 1*math.cos(-0.4)
        v0_y = 1*math.sin(-0.4)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, -0.4, 3537, 449)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([3563, 443]).cuda()
        v0_x = 1*math.cos(3.6)
        v0_y = 1*math.sin(3.6)
        new_input_data=add_new_agent(new_input_data,0.1, v0_x, v0_y, 3.6, 3588, 454.5)
        new_input_data['agent']["position"][-1,-1,:2]=torch.tensor([3570, 442]).cuda()
        new_input_data['agent']["position"][agent_index,-1,:2]=torch.tensor([3570, 442]).cuda()
        
    return new_input_data
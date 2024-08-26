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

import itertools
import multiprocessing
import shlex
from multiprocessing import Queue
import argparse
import random
import time


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--available_gpus",
    type=int,
    default=[6,7],
    nargs='+',
    help="List of available GPUs to use",
)

parser.add_argument(
    "--filename",
    type=str,
    help="Number of GPUs to use",
)


args, unknown_args = parser.parse_known_args()
print(args)
print(unknown_args)


def parse_user_defined_args(unknown_args):
    """Parses the user-defined arguments and returns a dictionary of the arguments and their values."""
    parsed_args = {}
    current_arg = None
    for arg in unknown_args:
        if arg.startswith('--'):
            # If the current argument starts with '-', it's a new user-defined argument
            current_arg = arg.lstrip('--')
            parsed_args[current_arg] = []
        else:
            # Otherwise, it's a value for the current user-defined argument
            if current_arg:
                parsed_args[current_arg].append(arg)
    return parsed_args


batch_size = 1
available_gpus = args.available_gpus
gpu_num = len(available_gpus)
worker_count = gpu_num * batch_size
gpu_queue = Queue()


def dry_run_a_py(filename, **args):
    additional_args = " ".join([f"--{key} {value}" for key, value in args.items()])
    command = f"python {filename} {additional_args}"
    print("Dry Run Command:", command)

def run_a_py(filename, **args):
    # Acquire a GPU from the queue
    gpu_id = gpu_queue.get()
    try:
        sleep_time = random.uniform(0, 1)
        time.sleep(sleep_time)
        additional_args = " ".join([f"--{key}={value}" for key, value in args.items()])
        command = f"python {filename} {additional_args}"
        import subprocess, os

        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        my_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(0.8)
        

        subprocess.run(shlex.split(command), check=True, env=my_env)
    finally:
        # Release the GPU back to the queue
        gpu_queue.put(gpu_id)


def product_dict(**kwargs):
  """
  Returns a list of dictionaries, where each dictionary is a product of the values
  in the input dictionary.

  Args:
    **kwargs: A dictionary of values.

  Returns:
    A list of dictionaries.
  """

  keys = kwargs.keys()
  for instance in itertools.product(*kwargs.values()):
    yield dict(zip(keys, instance))


def starmap_with_kwargs(pool, fn, filename, kwargs_iter):
    args_for_starmap = zip(itertools.repeat(fn), itertools.repeat(filename), kwargs_iter)
    pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, filename, kwargs):
    return fn(filename,**kwargs)


# Parse the unknown arguments using the custom function
parsed_args = parse_user_defined_args(unknown_args)
combinations = product_dict(**parsed_args)
print("Parsed user-defined arguments:", parsed_args)
print(combinations)
# Initialize the GPU queue with IDs of available GPUs
for index in available_gpus:
    for _ in range(batch_size):
        gpu_queue.put(index)

# Use dry run if specified, otherwise run each combination in parallel using multiprocessing
with multiprocessing.Pool(worker_count) as pool:
    # Use starmap to submit each combination to the worker pool
    starmap_with_kwargs(pool,run_a_py, args.filename, combinations)

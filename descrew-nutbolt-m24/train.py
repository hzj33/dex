# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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

import isaacgym

import os
import sys
import importlib
import hydra
import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from dexscrew.algo.ppo.ppo import PPO
from dexscrew.algo.student import (
    ProprioAdapt,
    PureBC,
    DiffusionLatentStudent,
    ConsistencyLatentStudent,
    ResidualConsistencyLatentStudent,
    FlowMatchingLatentStudent,
    BC,
    BCStudent,
    DAgger,
    DAggerStudent,
    DOTPG,
    DOTPGStudent,
)
from dexscrew.algo.ppo.diffusion_action_chunk_student import DiffusionActionChunkStudent
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict, print_dict
from dexscrew.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

# OmegaConf & Hydra Config
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_env
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


def import_wandb_package():
    """Import the real wandb package even if the repo has a local ./wandb run directory."""
    importlib.invalidate_caches()
    try:
        wandb = importlib.import_module("wandb")
        if hasattr(wandb, "init"):
            return wandb
    except Exception:
        pass

    cwd = os.path.abspath(os.getcwd())
    removed_entries = []
    sanitized_sys_path = []
    for entry in sys.path:
        normalized = os.path.abspath(entry or cwd)
        if normalized == cwd:
            removed_entries.append(entry)
            continue
        sanitized_sys_path.append(entry)

    original_sys_path = list(sys.path)
    sys.modules.pop("wandb", None)
    sys.path[:] = sanitized_sys_path
    try:
        importlib.invalidate_caches()
        wandb = importlib.import_module("wandb")
        if not hasattr(wandb, "init"):
            raise ImportError(
                "Imported module named 'wandb', but it does not expose wandb.init"
            )
        return wandb
    finally:
        sys.path[:] = original_sys_path
        importlib.invalidate_caches()


@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    cache_name = os.path.basename(config.train.ppo.output_name)
    if cache_name:
        run_name = f"{cache_name}_{time_str}"
    else:
        run_name = f"{config.wandb_name}_{time_str}"

    if config.checkpoint:
        if '*' in config.checkpoint:
            from glob import glob
            _ckpt = glob(config.checkpoint)
            assert len(_ckpt) == 1
            config.checkpoint = _ckpt[0]
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    cfg_dict = omegaconf_to_dict(config)
    print_dict(cfg_dict)

    config.seed = set_seed(config.seed)

    if config.wandb_activate:
        wandb = import_wandb_package()
        wandb_kwargs = dict(
            project=config.wandb_project,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )
        if config.wandb_group:
            wandb_kwargs["group"] = config.wandb_group
        if config.wandb_entity:
            wandb_kwargs["entity"] = config.wandb_entity
        run = wandb.init(**wandb_kwargs)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    output_dif = os.path.join('outputs', config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(config.train.algo)(env, output_dif, full_config=config)
    if config.test:
        assert config.train.load_path
        agent.restore_test(config.train.load_path)
        if bool(config.get("collect_rollout", False)):
            if not hasattr(agent, "collect_rollout"):
                raise ValueError(
                    f"{config.train.algo} does not support collect_rollout mode"
                )
            collect_steps = int(config.get("collect_steps", 256))
            collect_out = config.get("collect_out", "")
            collect_out = to_absolute_path(collect_out) if collect_out else None
            collect_save_point_cloud = bool(
                config.get("collect_save_point_cloud", True)
            )
            agent.collect_rollout(
                num_steps=collect_steps,
                save_path=collect_out,
                save_point_cloud=collect_save_point_cloud,
            )
        else:
            agent.test()
    else:
        date = str(datetime.datetime.now().strftime('%m%d%H'))
        skip_git_diff = os.getenv('DEXSCREW_SKIP_GIT_DIFF', '0').lower() in ('1', 'true', 'yes')
        if not skip_git_diff:
            print(git_diff_config('./'))
            gitdiff_suffix = ''
            os.system(f'git diff HEAD > {output_dif}/gitdiff{gitdiff_suffix}.patch')
        with open(os.path.join(output_dif, f'config_{date}_{git_hash()}.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        agent.restore_train(config.train.load_path)
        agent.train()


if __name__ == '__main__':
    main()

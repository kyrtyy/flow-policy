from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("bcmlp")
@dataclass
class BCMLPConfig(DiffusionConfig):
    """Plain behaviour cloning: Diffusion Policy's observation encoder followed by an MLP that regresses
    the whole action chunk. Only the vision and observation fields of the parent are used.
    """

    mlp_hidden_dim: int = 1024
    mlp_num_layers: int = 3

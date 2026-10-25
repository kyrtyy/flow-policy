from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from ..common import masked_mse
from .configuration_bcmlp import BCMLPConfig


class BCMLPModel(DiffusionModel):
    def __init__(self, config: BCMLPConfig):
        super().__init__(config)
        del self.unet
        # Same global-conditioning size DiffusionModel computes for its U-Net.
        cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            enc = self.rgb_encoder[0] if isinstance(self.rgb_encoder, nn.ModuleList) else self.rgb_encoder
            cond_dim += enc.feature_dim * len(config.image_features)
        if config.env_state_feature:
            cond_dim += config.env_state_feature.shape[0]

        dims = [cond_dim * config.n_obs_steps] + [config.mlp_hidden_dim] * config.mlp_num_layers
        layers = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), nn.Mish()]
        layers.append(nn.Linear(dims[-1], config.horizon * config.action_feature.shape[0]))
        self.mlp = nn.Sequential(*layers)

    def _predict(self, global_cond: Tensor) -> Tensor:
        return self.mlp(global_cond).view(global_cond.shape[0], self.config.horizon, -1)

    def conditional_sample(self, batch_size: int, global_cond: Tensor | None = None, **kwargs) -> Tensor:
        return self._predict(global_cond)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        pred = self._predict(self._prepare_global_conditioning(batch))
        return masked_mse(pred, batch[ACTION], batch, self.config.do_mask_loss_for_padding)


class BCMLPPolicy(DiffusionPolicy):
    config_class = BCMLPConfig
    name = "bcmlp"

    def __init__(self, config: BCMLPConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = BCMLPModel(config)
        self.reset()

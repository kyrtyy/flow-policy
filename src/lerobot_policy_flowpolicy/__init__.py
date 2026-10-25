# Importing the configs registers the "flow" and "bcmlp" policy types with LeRobot.
from lerobot.policies.diffusion import modeling_diffusion

from .bcmlp.configuration_bcmlp import BCMLPConfig
from .flow.configuration_flow import FlowConfig

__all__ = ["BCMLPConfig", "FlowConfig"]

# Fair few-step baseline for Diffusion Policy. diffusers defaults to "leading" spacing, so with 100 train
# steps K=1 samples only t=0 and K=2 samples [50, 0]: the first step treats pure noise as nearly clean.
# "trailing" gives [99] and [99, 49]. Training is unaffected (add_noise ignores spacing) and K=100 is
# identical either way. LeRobot's CLIs auto-import this plugin, so the patch covers lerobot-eval too.
# ponytail: monkeypatch; drop it if LeRobot exposes timestep_spacing in DiffusionConfig.
_make_noise_scheduler = modeling_diffusion._make_noise_scheduler
modeling_diffusion._make_noise_scheduler = lambda name, **kw: _make_noise_scheduler(
    name, timestep_spacing="trailing", **kw
)

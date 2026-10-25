import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_policy_flowpolicy.bcmlp.configuration_bcmlp import BCMLPConfig
from lerobot_policy_flowpolicy.bcmlp.modeling_bcmlp import BCMLPPolicy
from lerobot_policy_flowpolicy.flow.configuration_flow import FlowConfig
from lerobot_policy_flowpolicy.flow.modeling_flow import FlowModel, FlowPolicy

IMG = "observation.image"


def tiny(cfg_cls, **kw):
    # PushT shapes, shrunken network so this runs on CPU in seconds.
    return cfg_cls(
        input_features={
            IMG: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        horizon=16,
        n_action_steps=8,
        crop_shape=(84, 84),
        down_dims=(32, 64),
        pretrained_backbone_weights=None,
        use_group_norm=True,
        device="cpu",
        **kw,
    )


def batch(b=4, cfg=None):
    return {
        IMG: torch.randn(b, cfg.n_obs_steps, 3, 96, 96),
        OBS_STATE: torch.randn(b, cfg.n_obs_steps, 2),
        ACTION: torch.rand(b, cfg.horizon, 2) * 2 - 1,
        "action_is_pad": torch.zeros(b, cfg.horizon, dtype=torch.bool),
    }


@pytest.mark.parametrize("cfg_cls,policy_cls", [(FlowConfig, FlowPolicy), (BCMLPConfig, BCMLPPolicy)])
def test_train_step_and_rollout_shapes(cfg_cls, policy_cls):
    cfg = tiny(cfg_cls)
    policy = policy_cls(cfg)
    b = batch(cfg=cfg)
    loss, _ = policy.forward(b)
    assert torch.isfinite(loss)
    loss.backward()
    policy.eval()
    obs = {k: v for k, v in b.items() if k in (IMG, OBS_STATE)}
    assert policy.predict_action_chunk(obs).shape == (4, cfg.n_action_steps, 2)
    # select_action path used by lerobot-eval: one action per call, from a single-step observation.
    policy.reset()
    assert policy.select_action({k: v[:, -1] for k, v in obs.items()}).shape == (4, 2)


def test_diffusion_few_step_baseline_starts_from_noise():
    # Without trailing spacing, K=1 would denoise at t=0 and the diffusion baseline would be a strawman.
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

    for sched in ("DDPM", "DDIM"):
        m = DiffusionModel(tiny(DiffusionConfig, noise_scheduler_type=sched))
        m.noise_scheduler.set_timesteps(2)
        assert m.noise_scheduler.timesteps.tolist() == [99, 49]


@pytest.mark.parametrize(
    "kw",
    [{"num_inference_steps": k} for k in (1, 4, 10)]
    + [{"step_schedule": [0, 0.1, 1]}, {"num_inference_steps": 1, "warm_start": 0.5}],
)
def test_euler_sampler_hits_target_with_exact_velocity(kw):
    # Every demo equals `a`: the true flow velocity is (a - x) / (1 - t). The path is straight, so Euler
    # with any step count must land exactly on `a`. Catches a wrong time grid or integration direction.
    cfg = tiny(FlowConfig, **kw)
    model = FlowPolicy(cfg).diffusion
    a = torch.full((2, cfg.horizon, 2), 0.3)
    model._unet = lambda x, t, global_cond, d=None: (a - x) / (1 - t)[:, None, None]
    for _ in range(2):  # second call warm-starts from the first plan when warm_start > 0
        out = model.conditional_sample(2, noise=torch.randn(2, cfg.horizon, 2))
        torch.testing.assert_close(out, a)


def test_flow_loss_target_is_velocity():
    # A network that outputs the true conditional velocity x1 - x0 must get zero loss.
    cfg = tiny(FlowConfig)
    model = FlowPolicy(cfg).diffusion
    b = batch(cfg=cfg)
    b["observation.images"] = b[IMG].unsqueeze(2)
    # On the linear path, x1 - x0 = (x1 - x_t) / (1 - t).
    model._unet = lambda xt, t, global_cond, d=None: (b[ACTION] - xt) / (1 - t)[:, None, None]
    assert model.compute_loss(b).item() < 1e-8


def test_reflow_trains_toward_teacher_sample(tmp_path):
    # With a teacher, the regression target is the teacher's sample from the same noise, not the demo.
    teacher = FlowPolicy(tiny(FlowConfig))
    teacher.save_pretrained(tmp_path)
    cfg = tiny(FlowConfig, teacher_path=str(tmp_path), teacher_steps=2)
    model = FlowPolicy(cfg).diffusion
    b = batch(cfg=cfg)
    b["observation.images"] = b[IMG].unsqueeze(2)
    target = {}
    real = FlowModel._teacher_sample

    def spy(self, batch, x0):
        target["x1"] = real(self, batch, x0)
        return target["x1"]

    FlowModel._teacher_sample = spy
    try:
        model._unet = lambda xt, t, global_cond, d=None: (target["x1"] - xt) / (1 - t)[:, None, None]
        assert model.compute_loss(b).item() < 1e-8
        assert not torch.allclose(target["x1"], b[ACTION])
    finally:
        FlowModel._teacher_sample = real


def test_guide_decides_every_n_chunks_and_flow_continues_its_plan(tmp_path):
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    DiffusionPolicy(tiny(DiffusionConfig)).save_pretrained(tmp_path)
    cfg = tiny(FlowConfig, guide_path=str(tmp_path), guide_every=2, guide_steps=2, warm_start=0.5)
    policy = FlowPolicy(cfg)
    model = policy.diffusion
    b = batch(cfg=cfg)
    b["observation.images"] = b[IMG].unsqueeze(2)
    plan = torch.full((4, cfg.horizon, 2), 0.7)
    model.generate_actions(b)  # loads the guide
    model._guide.conditional_sample = lambda n, global_cond: plan
    model._unet = lambda x, t, global_cond, d=None: torch.zeros_like(x)  # flow keeps whatever it starts from
    policy.reset()
    out = [model.generate_actions(b) for _ in range(3)]
    torch.testing.assert_close(out[0], plan[:, 1:9])  # chunk 0: diffusion
    assert not torch.allclose(out[1], plan[:, 1:9])  # chunk 1: flow, half noise half plan
    torch.testing.assert_close(out[2], plan[:, 1:9])  # chunk 2: diffusion again


def test_refine_lands_on_diffusion_prediction(tmp_path):
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    DiffusionPolicy(tiny(DiffusionConfig)).save_pretrained(tmp_path)
    cfg = tiny(FlowConfig, refine_path=str(tmp_path), refine_steps=3, refine_scheduler="DDIM", num_inference_steps=1)
    model = FlowPolicy(cfg).diffusion
    b = batch(cfg=cfg)
    b["observation.images"] = b[IMG].unsqueeze(2)
    model.generate_actions(b)  # loads the refiner
    g, sched = model._refiner
    a = sched.alphas_cumprod
    target = torch.full((4, cfg.horizon, 2), 0.3)
    del g.unet  # oracle diffusion: its eps always points at `target`
    g.unet = lambda x, t, global_cond: (x - a[t[0]] ** 0.5 * target) / (1 - a[t[0]]) ** 0.5
    model._unet = lambda x, t, global_cond, d=None: torch.zeros_like(x)  # flow leaves pure noise
    out = model.generate_actions(b)
    torch.testing.assert_close(out, target[:, 1:9])


@pytest.mark.parametrize("kind,lo,hi", [("uniform", 0.45, 0.55), ("logit_normal", 0.45, 0.55), ("beta", 0.35, 0.45)])
def test_t_sampling_range_and_mean(kind, lo, hi):
    # beta: E[t] = 0.999 * (1 - 1.5 / 2.5) = 0.40, i.e. skewed toward the noisy end t = 0.
    model = FlowPolicy(tiny(FlowConfig, t_sampling=kind)).diffusion
    t = model._sample_t(20000, torch.zeros(1))
    assert 0 <= t.min() and t.max() < 1 and lo < t.mean() < hi


def test_shortcut_init_from_flow_matches_flow_and_loss_is_zero_on_straight_paths(tmp_path):
    flow = FlowPolicy(tiny(FlowConfig))
    flow.save_pretrained(tmp_path)
    sc = FlowPolicy(tiny(FlowConfig, shortcut=True, init_path=str(tmp_path)))
    b = batch(b=2, cfg=sc.config)
    b["observation.images"] = b[IMG].unsqueeze(2)
    x, t = torch.randn(2, 16, 2), torch.rand(2)
    with torch.no_grad():
        cond = flow.diffusion._prepare_global_conditioning(b)
        ref = flow.diffusion._unet(x, t, cond)
        for d in (None, torch.full((2,), 0.5)):  # zero-initialised step-size encoder: identical to flow at any d
            torch.testing.assert_close(sc.diffusion._unet(x, t, cond, d=d), ref)
    # A network giving the exact straight-path velocity satisfies both the flow and the consistency targets.
    model = sc.diffusion
    model._unet = lambda xt, t, global_cond, d=None: (b[ACTION] - xt) / (1 - t)[:, None, None]
    assert model.compute_loss(b).item() < 1e-8

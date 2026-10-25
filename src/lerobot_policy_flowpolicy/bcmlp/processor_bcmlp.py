from lerobot.processor import make_default_pre_post_processors


def make_bcmlp_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)

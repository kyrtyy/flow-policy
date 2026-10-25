from lerobot.processor import make_default_pre_post_processors


def make_flow_pre_post_processors(config, dataset_stats=None):
    return make_default_pre_post_processors(config, dataset_stats)

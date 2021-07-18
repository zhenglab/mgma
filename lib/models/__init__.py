# Code adapted from:
# https://github.com/facebookresearch/SlowFast

from .build import MODEL_REGISTRY, build_model  # noqa
from .mgma_builder import MGMANet  # noqa
from .mgma_shufflenet_builder import MGMAShuffleNet  # noqa

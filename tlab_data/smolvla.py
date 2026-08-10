"""SmolVLA feature schema for LIBERO."""

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def configure_for_libero(config: SmolVLAConfig) -> SmolVLAConfig:
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return config

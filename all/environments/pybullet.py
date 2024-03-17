from .gym import GymEnvironment


class PybulletEnvironment(GymEnvironment):
    short_names = {
        "ant": "AntBulletEnv-v0",
        "cheetah": "HalfCheetahBulletEnv-v0",
        "hopper": "HopperBulletEnv-v0",
        "humanoid": "HumanoidBulletEnv-v0",
        "walker": "Walker2DBulletEnv-v0",
    }

    def __init__(self, name, **kwargs):
        import pybullet_envs  # noqa: F401

        if name in self.short_names:
            name = self.short_names[name]
        super().__init__(name, legacy_gym=True, **kwargs)

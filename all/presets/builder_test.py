import unittest
from unittest.mock import Mock

from all.presets import PresetBuilder


class TestPresetBuilder(unittest.TestCase):
    def setUp(self):
        self.name = "my_preset"
        self.default_hyperparameters = {"lr": 1e-4, "gamma": 0.99}

        class MockPreset:
            def __init__(self, env, name, device, **hyperparameters):
                self.env = env
                self.name = name
                self.device = device
                self.hyperparameters = hyperparameters

        self.builder = PresetBuilder(
            self.name, self.default_hyperparameters, MockPreset
        )

    def test_default_name(self):
        agent = self.builder.env(Mock).build()
        self.assertEqual(agent.name, self.name)

    def test_override_name(self):
        agent = self.builder.name("cool_name").env(Mock).build()
        self.assertEqual(agent.name, "cool_name")

    def test_default_hyperparameters(self):
        agent = self.builder.env(Mock).build()
        self.assertEqual(agent.hyperparameters, self.default_hyperparameters)

    def test_override_hyperparameters(self):
        agent = self.builder.hyperparameters(lr=0.01).env(Mock).build()
        self.assertEqual(
            agent.hyperparameters, {**self.default_hyperparameters, "lr": 0.01}
        )

    def test_bad_hyperparameters(self):
        with self.assertRaises(KeyError):
            self.builder.hyperparameters(foo=0.01).env(Mock).build()

    def test_default_device(self):
        agent = self.builder.env(Mock).build()
        self.assertEqual(agent.device, "cuda")

    def test_override_device(self):
        agent = self.builder.device("cpu").env(Mock).build()
        self.assertEqual(agent.device, "cpu")

    def test_no_side_effects(self):
        self.builder.device("cpu").hyperparameters(lr=0.01).device("cpu").env(
            Mock
        ).build()
        my_env = Mock
        agent = self.builder.env(Mock).build()
        self.assertEqual(agent.name, self.name)
        self.assertEqual(agent.hyperparameters, self.default_hyperparameters)
        self.assertEqual(agent.device, "cuda")
        self.assertEqual(agent.env, my_env)

    def test_call_api(self):
        agent = (
            self.builder(device="cpu", hyperparameters={"lr": 0.01}, name="cool_name")
            .env(Mock)
            .build()
        )
        self.assertEqual(agent.name, "cool_name")
        self.assertEqual(
            agent.hyperparameters, {**self.default_hyperparameters, "lr": 0.01}
        )
        self.assertEqual(agent.device, "cpu")


if __name__ == "__main__":
    unittest.main()

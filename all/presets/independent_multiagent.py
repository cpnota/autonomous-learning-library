from .builder import PresetBuilder
from .preset import Preset
from all.agents import IndependentMultiagent
from all.logging import DummyLogger


class IndependentMultiagentPreset(Preset):
    def __init__(self, name, device, presets):
        super().__init__(name, device, presets)

    def agent(self, logger=DummyLogger(), train_steps=float('inf')):
        return IndependentMultiagent({
            agent_id: preset.agent(logger=logger, train_steps=train_steps)
            for agent_id, preset in self.hyperparameters.items()
        })

    def test_agent(self):
        return IndependentMultiagent({
            agent_id: preset.test_agent()
            for agent_id, preset in self.hyperparameters.items()
        })

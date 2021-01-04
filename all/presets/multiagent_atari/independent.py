from ..builder import preset_builder
from ..preset import Preset
from all.agents.multi.independent import IndependentMultiagent
from all.logging import DummyWriter


class IndependentMultiagentAtariPreset(Preset):
    def __init__(self, presets):
        self.presets = presets

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        return IndependentMultiagent({
            agent_id : preset.agent(writer=writer, train_steps=train_steps)
            for agent_id, preset in self.presets.items()
        })

    def test_agent(self):
        return IndependentMultiagent({
            agent_id : preset.test_agent()
            for agent_id, preset in self.presets.items()
        })

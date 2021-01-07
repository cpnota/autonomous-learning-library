from ..builder import preset_builder
from ..preset import Preset
from all.agents.multi.independent import IndependentMultiagent
from all.logging import DummyWriter


class ParameterSharingMultiagentPreset(Preset):
    def __init__(self, preset, agent_names):
        self.preset = preset
        self.agent_names = agent_names

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        base_agent = self.preset.agent(writer=writer, train_steps=train_steps)
        return IndependentMultiagent({
            agent_id : base_agent
            for agent_id in self.agent_names
        })

    def test_agent(self):
        base_agent = self.preset.test_agent()
        return IndependentMultiagent({
            agent_id : base_agent
            for agent_id in self.agent_names
        })

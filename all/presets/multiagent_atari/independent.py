from ..builder import preset_builder
from ..preset import Preset
from all.agents.multi.independent import IndependentMultiagent


class IndependentMultiagentAtariPreset(Preset):
    def __init__(self, env, builder):
        self.presets = {
            agent : builder.build() for agent in env.agents
        }
        self._agents = list(env.agents)

    def agent(self, writers=None, train_steps=float('inf')):
        return IndependentMultiagent({
            agent : self.presets[agent].agent(writer=writers[agent], train_steps=float('inf'))
        })

    def test_agent(self):
        return IndependentMultiagent({
            agent : self.presets[agent].test_agent()
        })

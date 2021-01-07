from ..builder import preset_builder
from ..preset import Preset
from all.agents.multi.independent import IndependentMultiagent
from all.logging import DummyWriter
from all.agents import Agent
import copy


def copy_agent(object):
    if isinstance(object, Agent):
        agent = copy.copy(object)
        for name, var in agent.__dict__.items():
            agent.__dict__[name] = copy_agent(var)
        return agent
    else:
        return object


class ParameterSharingMultiagentPreset(Preset):
    '''
    This is a hack that allows single player agents to be parameter shared policies.

    While it seems to work correctly for most agents in most circumstances,
    there is no guarentee and should only be used when the implications are understood.

    In particular, in the currently implementation schedules may not work as expected
    as the schedule count will be shared between agents.
    '''
    def __init__(self, preset, agent_names):
        self.preset = preset
        self.agent_names = agent_names

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        base_agent = self.preset.agent(writer=writer, train_steps=train_steps)
        # independently learn with shallow copies of the agents
        # the shallow copy ensures that agents use the same replay buffer and networks
        return IndependentMultiagent({
            agent_id : copy_agent(base_agent)
            for agent_id in self.agent_names
        })

    def test_agent(self):
        base_agent = self.preset.test_agent()
        return IndependentMultiagent({
            agent_id : copy_agent(base_agent)
            for agent_id in self.agent_names
        })

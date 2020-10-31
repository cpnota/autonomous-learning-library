from all.agents.multi.independent import IndependentMultiagent


def independent(agent_constructor):
    def _independent(env, writers=None):
        return IndependentMultiagent({
            agent : agent_constructor(env.subenvs[agent], writers[agent])
            for agent in env.agents
        })
    return _independent

__all__ = ["independent"]

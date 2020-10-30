from all.agents.multi.independent import IndependentMultiAgent


def independent(agent_constructor):
    def _independent(env, writers=None):
        return IndependentMultiAgent({
            agent : agent_constructor(env.subenvs[agent], writers[agent])
            for agent in env.agents
        })
    return _independent

__all__ = ["independent"]

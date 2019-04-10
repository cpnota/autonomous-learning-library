from all.experiments import Experiment

def run(agent_name, agent, env, episodes=None, frames=None):
    experiment = Experiment(
        env,
        frames=frames,
        episodes=episodes
    )
    experiment.run(agent, label=agent_name)

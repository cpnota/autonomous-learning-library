from .single_env_experiment import SingleEnvExperiment
from .parallel_env_experiment import ParallelEnvExperiment


def run_experiment(
        agents,
        envs,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        write_loss=True
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for agent in agents:
            make_experiment = get_experiment_type(agent)
            experiment = make_experiment(
                agent,
                env,
                logdir=logdir,
                quiet=quiet,
                render=render,
                write_loss=write_loss
            )
            experiment.train(frames=frames)
            experiment.test(episodes=test_episodes)


def get_experiment_type(agent):
    if is_parallel_env_agent(agent):
        return ParallelEnvExperiment
    return SingleEnvExperiment


def is_parallel_env_agent(agent):
    return isinstance(agent, tuple)

from all.presets import ParallelPreset

from .parallel_env_experiment import ParallelEnvExperiment
from .single_env_experiment import SingleEnvExperiment


def run_experiment(
    agents,
    envs,
    frames,
    logdir="runs",
    quiet=False,
    render=False,
    save_freq=100,
    test_episodes=100,
    verbose=True,
):
    if not isinstance(agents, list):
        agents = [agents]

    if not isinstance(envs, list):
        envs = [envs]

    for env in envs:
        for preset_builder in agents:
            preset = preset_builder.env(env).build()
            make_experiment = get_experiment_type(preset)
            experiment = make_experiment(
                preset,
                env,
                train_steps=frames,
                logdir=logdir,
                quiet=quiet,
                render=render,
                save_freq=save_freq,
                verbose=verbose,
            )
            experiment.save()
            experiment.train(frames=frames)
            experiment.save()
            experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment

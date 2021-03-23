from .single_env_experiment import SingleEnvExperiment
from .parallel_env_experiment import ParallelEnvExperiment
from all.presets import ParallelPreset


def run_experiment(
        agents,
        envs,
        frames,
        logdir='runs',
        quiet=False,
        render=False,
        test_episodes=100,
        write_loss=True,
        writer="tensorboard"
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
                write_loss=write_loss,
                writer=writer
            )
            experiment.train(frames=frames)
            experiment.save()
            experiment.test(episodes=test_episodes)
            experiment.close()


def get_experiment_type(preset):
    if isinstance(preset, ParallelPreset):
        return ParallelEnvExperiment
    return SingleEnvExperiment

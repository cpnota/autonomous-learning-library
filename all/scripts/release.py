'''Create slurm tasks to run release test suite'''
from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SlurmExperiment
from all.presets import atari, classic_control, continuous


def main():
    # run on gpu
    device = 'cuda'

    def get_agents(preset):
        agents = [getattr(preset, agent_name) for agent_name in preset.__all__]
        return [agent(device=device) for agent in agents]

    SlurmExperiment(
        get_agents(atari),
        AtariEnvironment('Breakout', device=device),
        10e7,
        sbatch_args={
            'partition': '1080ti-long'
        }
    )

    SlurmExperiment(
        get_agents(classic_control),
        GymEnvironment('CartPole-v0', device=device),
        100000,
        sbatch_args={
            'partition': '1080ti-short'
        }
    )

    SlurmExperiment(
        get_agents(continuous),
        GymEnvironment('LunarLanderContinuous-v2', device=device),
        500000,
        sbatch_args={
            'partition': '1080ti-short'
        }
    )


if __name__ == "__main__":
    main()

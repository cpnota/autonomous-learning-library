import argparse
import os
import subprocess
import sys
from .run_experiment import run_experiment


# track the number of experiments created
# in the current process
ID = {
    "value": 1
}


class SlurmExperiment:
    def __init__(
            self,
            agents,
            envs,
            frames,
            test_episodes=100,
            write_loss=False,
            job_name='autonomous-learning-library',
            script_name='experiment.sh',
            outdir='out',
            logdir='runs',
            sbatch_args=None,
    ):
        if not isinstance(agents, list):
            agents = [agents]

        if not isinstance(envs, list):
            envs = [envs]

        self.agents = agents
        self.envs = envs
        self.frames = frames
        self.test_episodes = test_episodes
        self.write_loss = write_loss
        self.job_name = job_name
        self.script_name = script_name
        self.outdir = outdir
        self.logdir = logdir
        self.sbatch_args = sbatch_args or {}
        self.parse_args()

        # handles multiple experiments created in single script
        self._id = ID["value"]
        ID["value"] += 1

        if self.args.experiment_id:
            # We are inside a slurm task.
            # Only run the experiment if the ID matches.
            if self._id == self.args.experiment_id:
                self.run_experiment()
        else:
            # otherwise, we need to create the
            # bash file and call sbatch
            self.queue_jobs()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Run an Atari benchmark.')
        parser.add_argument('--experiment_id', type=int)
        self.args = parser.parse_args()

    def run_experiment(self):
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        env = self.envs[int(task_id / len(self.agents))]
        agent = self.agents[task_id % len(self.agents)]
        run_experiment(
            agent,
            env,
            self.frames,
            test_episodes=self.test_episodes,
            logdir=self.logdir,
            write_loss=self.write_loss
        )

    def queue_jobs(self):
        self.create_sbatch_script()
        self.make_output_directory()
        self.run_sbatch_script()

    def create_sbatch_script(self):
        script = open(self.script_name, 'w')
        script.write('#!/bin/sh\n\n')
        num_experiments = len(self.envs) * len(self.agents)

        sbatch_args = {
            'job-name': self.job_name,
            'output': os.path.join(self.outdir, 'all_%A_%a.out'),
            'error': os.path.join(self.outdir, 'all_%A_%a.err'),
            'array': '0-' + str(num_experiments - 1),
            'partition': '1080ti-short',
            'ntasks': 1,
            'mem-per-cpu': 4000,
            'gres': 'gpu:1'
        }
        sbatch_args.update(self.sbatch_args)

        for key, value in sbatch_args.items():
            script.write('#SBATCH --' + key + '=' + str(value) + '\n')
        script.write('\n')

        script.write('python ' + sys.argv[0] + ' --experiment_id ' + str(self._id) + '\n')
        script.close()
        print('created sbatch script:', self.script_name)

    def make_output_directory(self):
        try:
            os.mkdir(self.outdir)
            print('Created output directory:', self.outdir)
        except FileExistsError:
            print('Output directory already exists:', self.outdir)

    def run_sbatch_script(self):
        result = subprocess.run(
            ['sbatch', self.script_name],
            stdout=subprocess.PIPE,
            check=True
        )
        print(result.stdout.decode('utf-8').rstrip())

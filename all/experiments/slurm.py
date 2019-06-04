import argparse
import os
import subprocess
import sys
from .experiment import Experiment

SCRIPT_NAME = 'experiment.sh'
OUT_DIR = 'out'

class SlurmExperiment:
    def __init__(
            self,
            agent,
            envs,
            frames,
            job_name='autonomous-learning-library',
            hyperparameters=None,
            sbatch_args=None,
    ):
        self.agent = agent
        self.envs = envs
        self.frames = frames
        self.job_name = job_name
        self.hyperparameters = hyperparameters
        self.sbatch_args = sbatch_args
        self.parse_args()

        if self.args.reentrant:
            # we are in a call to sbatch
            self.run_experiment()
        else:
            # otherwise, we need to create the
            # bash file and call sbatch
            self.queue_jobs()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Run an Atari benchmark.')
        parser.add_argument('--reentrant', dest='reentrant', action='store_true')
        parser.set_defaults(reentrant=False)
        self.args = parser.parse_args()

    def run_experiment(self):
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        env = self.envs[index]
        experiment = Experiment(env, frames=self.frames)
        experiment.run(self.agent(self.hyperparameters))

    def queue_jobs(self):
        self.create_sbatch_script()
        self.make_output_directory()
        self.run_sbatch_script()

    def create_sbatch_script(self):
        script = open(SCRIPT_NAME, 'w')
        script.write('#!/bin/sh\n\n')

        sbatch_args = {
            'job-name': self.job_name,
            'output': 'out/all_%A_%a.out',
            'error': 'out/all_%A_%a.err',
            'array': '0-' + str(len(self.envs) - 1),
            'partition': '1080ti-short',
            'ntasks': 1,
            'mem-per-cpu': 4000
        }

        for key, value in sbatch_args.items():
            script.write('#SBATCH --' + key + '=' + str(value) + '\n')
        script.write('\n')

        script.write('python ' + sys.argv[0] + ' --reentrant\n')
        script.close()
        print('created sbatch script:', SCRIPT_NAME)

    def make_output_directory(self):
        try:
            os.mkdir(OUT_DIR)
            print('Created output directory:', OUT_DIR)
        except FileExistsError:
            print('Output directory already exists:', OUT_DIR)

    def run_sbatch_script(self):
        result = subprocess.run(
            ['sbatch', SCRIPT_NAME],
            stdout=subprocess.PIPE
        )
        print(result.stdout.decode('utf-8').rstrip())

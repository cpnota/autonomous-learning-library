import argparse
import sys

class SlurmExperiment:
    def __init__(
            self,
            agent,
            envs,
            job_name='autonomous-learning-library',
            hyperparameters=None,
            sbatch_args=None,
    ):
        self.agent = agent
        self.envs = envs
        self.job_name = job_name
        self.hyperparameters = hyperparameters
        self.sbatch_args = sbatch_args
        self.parse_args()

        # we are in a call to sbatch
        if self.args.reentrant:
            self.run_trial()
        else:
            # otherwise, we need to create the
            # bash file and call sbatch
            self.queue_jobs()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Run an Atari benchmark.')
        parser.add_argument('--reentrant', help='True if we are launching a script')
        self.args = parser.parse_args()
    
    def run_trial(self):
        print('ran trial')
        return # TODO

    def queue_jobs(self):
        script = open("slurm_experiment.sh", "w")
        script.write('#!/bin/sh\n\n')

        sbatch_args = {
            'job-name': self.job_name,
            'output': 'all_%A_%a.out',
            'error': 'all_%A_%a.err',
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

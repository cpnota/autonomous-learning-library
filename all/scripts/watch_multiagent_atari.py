import argparse
import time
import torch
from all.bodies import DeepmindAtariBody
from all.environments import MultiagentAtariEnv
from all.experiments import load_and_watch


def watch(env, filename, fps, reload):
    agent = torch.load(filename).test_agent()

    while True:
        watch_episode(env, agent, fps)
        if reload:
            try:
                agent = torch.load(filename).test_agent()
            except Exception as e:
                print('Warning: error reloading model: {}'.format(filename))
                print(e)


def watch_episode(env, agent, fps):
    env.reset()
    for _ in env.agent_iter():
        env.render()
        state = env.last()
        action = agent.act(state)
        if state.done:
            env.step(None)
        else:
            env.step(action)
        time.sleep(1 / fps)


def main():
    parser = argparse.ArgumentParser(description="Watch pretrained multiagent atari")
    parser.add_argument("env", help="Name of the Atari game (e.g. pong-v1)")
    parser.add_argument("filename", help="File where the model was saved.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="Playback speed",
    )
    parser.add_argument(
        "--reload",
        action="store_true", default=False, help="Reload the model from disk after every episode"
    )
    args = parser.parse_args()
    env = MultiagentAtariEnv(args.env, device=args.device)
    watch(env, args.filename, args.fps, args.reload)


if __name__ == "__main__":
    main()

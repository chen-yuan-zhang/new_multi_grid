from multigrid.envs.goal_prediction import AGREnv
from .gr_pursuer.agents.target import AstarTarget,OnlineAstarTarget
from .gr_pursuer.agents.observer import BeliefUpdateObserver,Observer
from .core.actions import Action


from matplotlib import pyplot as plt

env = AGREnv(size = 15,num_goals = 3, render_mode='human')

observation, info = env.reset()
TargetAgent = OnlineAstarTarget(env)
ObserverAgent = BeliefUpdateObserver(env)


while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    obs_action = ObserverAgent.compute_action(observation[0])
    actions[1] = TargetAgent.compute_action(observation)

    img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
    observation, reward, terminated, truncated, info = env.step(actions)
    input()

env.close()
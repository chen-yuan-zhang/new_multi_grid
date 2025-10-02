from multigrid.envs.branchmark import AGREnv
from .agents.target import eGreedyTarget
from matplotlib import pyplot as plt

env = AGREnv(size = 15,num_goals = 5, render_mode='human')



observation, info = env.reset()
TargetAgent = eGreedyTarget(env)
# print(observation)
# print(info)

while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    actions[1] = TargetAgent.compute_action(observation)
    img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
    #print(img.shape)
    observation, reward, terminated, truncated, info = env.step(actions)
    #print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")
    input()
    
env.close()
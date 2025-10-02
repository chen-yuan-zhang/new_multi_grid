from .gr_pursuer.agents.target import AstarTarget,eGreedyTarget
from .gr_pursuer.agents.observer import BeliefUpdateObserver

from multigrid.envs.benchmark import AGREnv

from matplotlib import pyplot as plt

#config = "run1_config.json"
config = None

env = AGREnv(config=config, render_mode='human')


observation, info = env.reset()
TargetAgent = eGreedyTarget(env)
ObserverAgent = BeliefUpdateObserver(env)

print(observation)
# print(info)
if config != None:
    TargetAgent_actions = env.TargetAgent_actions
    steps = 0
    while not env.unwrapped.is_done():
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        actions[0] = ObserverAgent.compute_action(observation[0])
        actions[1] = TargetAgent_actions[steps]
        img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
        observation, reward, terminated, truncated, info = env.step(actions)
        input()
        steps += 1
    
    env.save_config("run1_config.json", extra={"TargetAgent_actions": TargetAgent_actions})
    env.close()

else:
    TargetAgent_actions = []
    while not env.unwrapped.is_done():
        actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
        actions[0] = ObserverAgent.compute_action(observation[0])
        actions[1] = TargetAgent.compute_action(observation)
        img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
        #print(img.shape)
        observation, reward, terminated, truncated, info = env.step(actions)
        TargetAgent_actions.append(actions[1])
        #print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")
        input()
        
    env.save_config("run1_config.json", extra={"TargetAgent_actions": TargetAgent_actions})
    env.close()

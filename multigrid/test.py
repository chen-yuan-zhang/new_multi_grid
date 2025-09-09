from multigrid.envs.goal_prediction import AGREnv
from .gr_pursuer.agents.target import AstarTarget
from .gr_pursuer.agents.observer import BeliefUpdateObserver,Observer
from .core.actions import Action


from matplotlib import pyplot as plt

env = AGREnv(size = 12,num_goals = 3, render_mode='human')



observation, info = env.reset()
TargetAgent = AstarTarget(env)
#ObserverAgent = Observer(env)
ObserverAgent = BeliefUpdateObserver(env)

# print(observation)
# print(info)

# while not env.unwrapped.is_done():
#     actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
#     actions[1] = TargetAgent.compute_action(observation)
#     actions[0] = ObserverAgent.compute_action(observation[0])

#     img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
#     #print(img.shape)
#     observation, reward, terminated, truncated, info = env.step(actions)
#     #print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")
#     input()
    
# env.close()


while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}

    # ObserverAgent
    observer = env.unwrapped.agents[0]
    obs_action = ObserverAgent.compute_action(observation[0])
    fwd_pos = observer.front_pos
    fwd_obj = env.grid.get(*fwd_pos)
    print(fwd_obj)
    if fwd_obj is not None:
        actions[0] = Action.remove   # 👈 如果前方不是空，就remove
    else:
        actions[0] = obs_action

    # TargetAgent（照旧）
    actions[1] = TargetAgent.compute_action(observation)

    # 渲染、执行
    img = env.grid.render(tile_size=32, agents=env.unwrapped.agents, highlight_mask=None)
    observation, reward, terminated, truncated, info = env.step(actions)
    input()

env.close()
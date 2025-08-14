from multigrid.envs.goal_prediction import AGREnv

env = AGREnv(render_mode='human')

observation, info = env.reset()
print(observation)
print(info)

while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    observation, reward, terminated, truncated, info = env.step(actions)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")
    input()
    
env.close()
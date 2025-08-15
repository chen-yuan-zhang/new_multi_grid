from multigrid.envs.goal_prediction import AGREnv
from .agents.target import Target

env = AGREnv(render_mode='human')

TargetAgent = Target(env)

observation, info = env.reset()
print(observation)
print(info)

while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    actions[1] = TargetAgent.compute_action(observation)
    observation, reward, terminated, truncated, info = env.step(actions)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")
    input()
    
env.close()
from multigrid.envs.goal_prediction import AGREnv
from .agents.target import AstarTarget
from matplotlib import pyplot as plt

env = AGREnv(render_mode='human')



observation, info = env.reset()
TargetAgent = AstarTarget(env)
print(observation)
print(info)

while not env.unwrapped.is_done():
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    actions[1] = TargetAgent.compute_action(observation)
    img = env.grid.render(tile_size=32, agents=env.unwrapped.agents[1:], highlight_mask=None)
    # visualize 
    plt.imshow(img)
    plt.show()
    plt.close()
    input()

    observation, reward, terminated, truncated, info = env.step(actions)
    print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}")

    
    
env.close()
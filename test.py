from multigrid.envs.new_locked import AGRlocked
import multigrid.envs

env = AGRlocked(render_mode='human')
observations, infos = env.reset()

while not env.unwrapped.is_done():
   # this is where you would insert your policy / policies
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    # print("rewards: ",rewards)
    # print("terminations: ",terminations)
    # print("truncations: ", truncations)
    # print("infos: ",infos)
    input()
    
env.close()
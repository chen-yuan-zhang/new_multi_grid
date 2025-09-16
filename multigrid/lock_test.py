from multigrid.envs.new_locked import AGRlocked
from .gr_pursuer.agents.lock_target import LockTarget
from .gr_pursuer.agents.new_KD_obs import BeliefUpdateObserver

import multigrid.envs

env = AGRlocked(render_mode='human')
observations, infos = env.reset()
TargetAgent = LockTarget(env)
ObserverAgent = BeliefUpdateObserver(env)


while not env.unwrapped.is_done():
   # this is where you would insert your policy / policies
    actions = {agent.index: agent.action_space.sample() for agent in env.unwrapped.agents}
    actions[1] = TargetAgent.compute_action(observations,env)
    actions[0] = ObserverAgent.compute_action(observations[0])

    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    input()
env.close()
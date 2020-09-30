from armEnv import ArmEnv
from rl import DDPG

import matplotlib.pyplot as plt
Dict = {}


MAX_EPISODES = 1000 
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []

#Train function    
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))            
                # adds the data for successful episode vs reward training in Dict
                if done:
                    Dict[i] = ep_r 
                else:
                    Dict[i] = 0.
                        
                break
    rl.save()

# Evalute    
def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)
        
        
#Plot 
def plot(Dict, MAX_EPISODES):
    # Plot
    lists = sorted(Dict.items())
    x, y = zip(*lists)
    plt.plot(x,y)
    plt.xlabel('EPISODES')
    plt.ylabel('REWARDS')
    plt.title('Training Success') 
    plt.show()

while(ON_TRAIN):
    #training starts
    train()
    # Plots successfull episodes vs reward occured during training
    plot(Dict, MAX_EPISODES)
    # Evaluate
    eval()


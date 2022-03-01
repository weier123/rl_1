#main.py
# 导入环境和学习方法
from part1.env import ArmEnv
from part1.rl import DDPG
#设置全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 200

#设置环境
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound  = env.action_bound

#设置学习方法（这里使用DDPG）
rl = DDPG(a_dim, s_dim, a_bound)

#开始训练
for i in range(MAX_EPISODES):
    s = env.reset()   #初始化回合设置
    for j in range(MAX_EP_STEPS):
        env.render() #环境的渲染
        a = rl.choose_action(s) #RL选择动作
          
import torch
import numpy as np
from model import ReplayMemory, Agent, convert_np
from main import args


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
memory = ReplayMemory(args.memory, args.batch_size)
agent = Agent(memory, 128, 3)

def train(env, render_flag):
    done = False
    score = 0

    frame = env.reset()
    frame = convert_np(frame)
    
    while not done:

        # Epsilon greedy
        action = agent.take_action(frame)
        # Action += 1, because we avoid 0 action (noop)
        frame_, reward, done, info = env.step(action+1)
        
        if render_flag:
            env.render()
        
        # Torch Tensor
        frame_ = convert_np(frame_)
        score += reward       

        # reward max is 1.0, useful in Bellman equation
        reward = max(min(reward, 1.0), -1.0)
        
        agent.memory.push_memory(frame, action, reward, frame_)
        
        frame = frame_
        
        if agent.should_learn():
            agent.learn()
        
    return score, agent.epsilon
        
        
def test(env, render_flag):
    done = False
    score = 0
    agent.eval()
    nothing = 0
    # If fire is True, then
    # FIREEE !!
    fire = False
    
    with torch.no_grad():

        frame = env.reset()
        env.step(1)
        frame = convert_np(frame)

        while not done:

            action = agent.take_action(frame)
            # Check fire button
            action =  action if not fire else 0
            
            frame_, reward, done, info = env.step(action+1)

            if reward == 0:
                nothing += 1
                
            else:
                # AI play the game with the ball
                nothing = 0
                
            if fire:
                fire = False
                
            if nothing >= 600:
                # If the AI, don't press fire button
                # We press it
                fire = True
            
            if render_flag:
                env.render()
            
            frame_ = convert_np(frame_)
            score += reward
            
            frame = frame_

    return score
        

import gym
import torch
import argparse
import gym.wrappers as wrappers
from pathlib import Path
from train import *
import sys


parser = argparse.ArgumentParser()

# Hyper Parameters
parser.add_argument('--epsilon', type=float, 
                        help='Set the start value of epsilon', default=1.0)
parser.add_argument('--min_epsilon', type=float, 
                        help='End of epsilon value', default=0.01)
parser.add_argument('--memory', type=int, 
                        help='Memory size', default=100000)
parser.add_argument('--batch_size', type=int, 
                        help='Batch size', default=32)
parser.add_argument('--gamma', type=float, 
                        help='Gamma value', default=0.99)
parser.add_argument('--learning_rate', type=float, 
                        help='Learning rate value', default=0.0005)
parser.add_argument('--episodes', type=int, 
                        help='Number of episode', default=10000)
parser.add_argument('--record', action='store_true', 
                        help='Record boolean')
parser.add_argument('--test', type=str, 
                        help='Set model to test it')
parser.add_argument('--render', action='store_true', 
                        help='Render Boolean')

args = parser.parse_args()

if __name__ == '__main__':
    
    history = []
    highest_score = 0
    score = 0
    
    env = gym.make('Breakout-ram-v0')
    env.reset()
    
    # Save videos
    if args.record:
        env = wrappers.Monitor(env, "records/Breakout", video_callable=lambda episode_id:True, force=True)
    
    if not args.test:
        # Training
        for episode in range(1, args.episodes):
            reward, epsilon = train(env, args.render)
            print('EPISODE {0} REWARD [{1}] EPSILON [{2}]'.format(episode, reward, epsilon))
            history.append([episode, reward])
            agent.plot(history)
            agent.update(episode)
            
            # If it's the highest score, save model && clear memory
            if reward > highest_score:
                highest_score = reward
                print('Highest SCORE [{0}]'.format(highest_score))
                agent.memory.clear()
                agent.save()
    
    else:
        # Try to load model
        file = Path(args.test)
        
        if file.is_file():
            agent.load(args.test)
            
        else:
            print('File not found')
            sys.exit(1)
            
    highest_score = 119
    history = []
    
    # Eval Model
    for episode in range(1, args.episodes):
    
        reward = test(env, args.render)
        print('EPISODE {0} REWARD [{1}]'.format(episode, reward))

        history.append([episode, reward])
        agent.plot(history)
        
        # Try to beat highest score
        if reward > highest_score:
            print('HIGHEST SCORE [{0}]'.format(reward))
            break
        
    print('Closing environnement..')
    env.close()

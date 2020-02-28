# DQN-Breakout-using-Pytorch
This project, is an **AI**, using openai/gym to play ***Breakout*** <br />
Requirements:
```shell
python3 install -r requirements.txt
``` 
<br /> <br />
![alt text](/records/DQN_Breakout.gif)

## **LEARN**: <br />
  ```shell
  python3 main.py
  ```
  You can set all the parameters <br />
  `--epsilon` `--min_epsilon` <br />
  `--memory` `--batch_size` <br />
  `--gamma` `--learning_rate` <br />
  `--episodes` `--record` <br />
  `--test` `--saved_as` <br />
  `--render` <br />
  
## **EVAL**: <br />
  To view test the model just put this: <br />
  ```shell
  python3 main.py --test YOUR_MODEL.pth
  ```
  
  Obviously, in the folder models, there are **trained model** <br />
  *PS*: Try them
  
  <br />
  <br />
  
### **DEFAULT HYPER PARAMETERS**:
  ```python3
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
                          help='Learning rate value', default=0.0003)
  parser.add_argument('--episodes', type=int, 
                          help='Number of episode', default=10000)
  parser.add_argument('--record', action='store_true', 
                          help='Record boolean')
  parser.add_argument('--test', type=str, 
                          help='Set model to test it', default='Breakout.pth)
  parser.add_argument('--saved_as', type=str, 
                          help='Name to save it', default='Breakout.pth)
  parser.add_argument('--render', action='store_true', 
                          help='Render Boolean')
 ```
<br />

### **More info**:
  I've used **Deep Q Learning**, to train the model with totally **off-policy** using Replay Memory <br />
  In the learn function, I used *bellman equation* <br />
  `REWARDS + GAMMA * Q_NEXT`
  to set the target value, <br />
  Thanks for watching my github page,

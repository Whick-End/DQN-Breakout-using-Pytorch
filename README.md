# DQN-Breakout-using-Pytorch
This project, is an **AI**, using openai/gym to play ***Breakout*** <br />
Requirements:
```shell
python3 install -r requirements.txt
``` 
<br /> <br />
![alt text](/records/DQN_Breakout.gif)

-**LEARN**: <br />
  ```shell
  python3 main.py
  ```
  You can set all the parameters <br />
  `--epsilon` `--min_epsilon` <br />
  `--memory` `--batch_size` <br />
  `--gamma` `--learning_rate` <br />
  `--episodes` `--record` <br />
  `--test` `render` <br />
  
-**EVAL**: <br />
  To view test the model just put this: <br />
  ```shell
  python3 main.py --test NN_Saved.pth
  ```
  <br />

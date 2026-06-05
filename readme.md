This repository trains a velocity tracking policy on a Unitree Go2 Quadruped using a home-made PPO algorithm.
The robot model is from [mujoco model gallery](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2)

Training: 
```python
python train_self.py train
```
(I divided the training in to multiple stages, you might want to comment/uncomment some.)

Visualization: 
```python
mjpython train_self.py test
```


After 5e6 env steps of training, velocity tracking behavior should appear
After 1e7 env steps of training, gait emerges
After 2e7 env steps of training, gait should be clean, and reward is about 28
(The home-made ppo might print out meaningless statistics before all environments complete at least 1 episode. You might see 0 epi length/reward at the begining of each training stage.)

It takes ~4 hours for 2e7 env steps of training on a M1 Macbook Pro.


<p>
  <img width="100%" src="https://github.com/bstars/Go2Track/blob/main/record.gif">
</p>



Zero-shot navigation using GPT:
<p>
  <img width="100%" src="https://github.com/bstars/Go2Track/blob/main/navigate.png">
</p>

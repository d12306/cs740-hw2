# cs740-hw2
Implementation of HW2 for CS740 following the [paper](https://dl.acm.org/doi/pdf/10.1145/3098822.3098843).

## Installation

`pip install -r requirements.txt`

## Train the ABR algorithm by A3C

```python
python train_a3c.py
```
## Train the ABR algorithm by PPO

```python
python train_ppo.py
```
## Test the model from pretrained A3C RL agent

```python
python a3c_test.py ./models_a3c/pretrain_linear_reward.ckpt
```

## Test the model from pretrained PPO RL agent

```python
python test.py ./models_ppo/nn_model_ep_151200.ckpt
```

## Plot the comparison results

```python
python plot.py
```
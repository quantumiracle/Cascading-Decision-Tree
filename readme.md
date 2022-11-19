# Cascading Decision Tree (CDT) for Explainable Reinforcement Learning

Open-source code for paper *CDT: Cascading Decision Trees for Explainable Reinforcement Learning* (https://arxiv.org/abs/2011.07553).

Data folder: the data folders (`data` and `data_no_norm`) should be put at the root of the repo to run the code. See issues:  [#4](https://github.com/quantumiracle/Cascading-Decision-Tree/issues/4). The data folders are stored at the [Google Drive](https://drive.google.com/drive/folders/18GGBNZhugIAQXJ1TXtwJBwI6HkDWIJay?usp=sharing).

# File Structure

* data: all data for experiments (not maintained in the repo, but can be collected with the given scripts below)
  * mlp: data for MLP model;
  * cdt: data for CDT model;
  * sdt: data for SDT model;
  * il: data for general Imitation Learning (IL);
  * rl: data for general Reinforcement Learning (RL);
  * cdt_compare_depth: data for cdt with different depths in RL;
  * sdt_compare_depth: data for sdt with different depths in RL;
* src: source code
  * mlp: training configurations for MLP as policy function approximator;
  * cdt: the Cascading Decision Tree (CDT) class and necessary functions;
  * sdt: the Soft Decision Tree (SDT) class and necessary functions;
  * hdt: the heuristic agents;
  * il: configurations for Imitation Learning (IL);
  * rl: configurations for Reinforcement Learning (RL) and RL agents (e.g., PPO) etc;
  * utils: some common functions
  * `il_data_collect.py`: collect dataset (state-action from heuristic or well-trained policy) for IL;
  * `rl_data_collect.py`: collect dataset (states during training for calculating normalization statistics) for RL;
  * `il_train.py`: train IL agent with different function approximators (e.g., SDT, CDT);
  * `rl_train.py`: train RL agent different function approximators (e.g., SDT, CDT, MLP);
  * `il_eval.py`: evaluate the trained IL agents before and after tree discretization, based on prediction accuracy;
  * `rl_eval.py`: evaluate the trained RL agents before and after tree discretization, based on episodic reward;
  * `il_train.sh`: bash to run IL  test with different models on server;
  * `rl_train.sh`: bash to run RL test with different models on server;
  * `rl_train_compare_sdt.py`: train RL agent with SDT;
  * `rl_train_compare_cdt.py`: train RL agent with SDT;
  * `rl_train_compare_sdt.sh`: bash to run RL test with SDT of different depths on server;
  * `rl_train_compare_cdt.sh`: bash to run RL test with CDT of different depths on server;
* visual
  * `plot.ipynb`: plot learning curves, etc.
  * `params.ipynb`: quantitive analysis of model parameters (SDT and CDT).
  * `stability_analysis.ipynb`: refer to the stability analysis in paper--compare the tree weights.

# To Run

For fully replicating the experiments in the paper, the code needs to run in several stages.

### A. Reinforcement Learning Comparison with SDT, CDT and MLP

1. Collect dataset: for state normalization

   ``` bash
   cd ./src
   python rl_data_collect.py
   ```

2. Get statistics on dataset

   ````bash
   cd rl
   jupyter notebook
   ````

   open `stats.ipynb` and run cells in it to generate files for dataset statistics.
   
   Step 1, 2 can be skipped is not using state normalization. 

3. Train RL agents with different policy function approximators: SDT, CDT, MLP

   ```bash
   cd ..
   python rl_train.py --train --env='CartPole-v1' --method='sdt' --id=0
   python rl_train.py --train --env='LunarLander-v2' --method='cdt' --id=0
   python rl_train.py --train --env='MountainCar-v0' --method='mlp' --id=0
   ```

   or simply run with:

   ````bash
   ./rl_train.sh
   ````

4. Evaluate the trained agents (with discretization operation)

   ````bash
   python rl_eval.py --env='CartPole-v1' --method='sdt'
   python rl_eval.py --env='LunarLander-v2' --method='cdt'
   ````


5. Results visualization

   ```bash
   cd ../visual
   jupyter notebook
   ```

   see in `plot.ipynb`.

### B. Imitation Learning Comparison with SDT and CDT

1. Collect dataset: for (1) state normalization and (2) as imitation learning dataset

   ```bash
   cd ./src
   python il_data_collect.py
   ```

2. Train RL agents with different policy function approximators: SDT, CDT

   ```bash
   python il_train.py --train --env='CartPole-v1' --method='sdt' --id=0
   python il_train.py --train --env='LunarLander-v2' --method='cdt' --id=0
   ```

   or simply run with:

   ```bash
   ./il_train.sh
   ```

3. Evaluate the trained agents

   ```bash
   python il_eval.py --env='CartPole-v1' --method='sdt'
   python il_eval.py --env='LunarLander-v2' --method='cdt'
   ```

4. Results visualization

   ```
   cd ../visual
   jupyter notebook
   ```

   see in `plot.ipynb`.
   
### B'. Imitation Learning with DAGGER and Q-DAGGER
DAGGER and Q-DAGGER methods in [VIPER](https://arxiv.org/abs/1805.08328) are compared in the paper as well under the imitation learning setting. Code in `./src/viper/`. Credit gives to [Hangrui (Henry) Bi
](https://github.com/20171130).

### C. Tree Depths for SDT and CDT in Reinforcement Learning

 Run the comparison with different tree depths:

For SDT:

```bash
./rl_train_compare_sdt.sh
```

For CDT:

```
./rl_train_compare_cdt.sh
```

### D. Stability Analysis

Compare the tree weights of different agents in IL:

```bash
cd ./visual
jupyner notebook
```

See in `stability_analysis.ipynb`.

### E. Model Simplicity

Quantitative analysis of number of model parameters:

```bash
cd ./visual
jupyter notebook
```

See in `params.ipynb`.

## Citation:
```
@article{ding2020cdt,
  title={Cdt: Cascading decision trees for explainable reinforcement learning},
  author={Ding, Zihan and Hernandez-Leal, Pablo and Ding, Gavin Weiguang and Li, Changjian and Huang, Ruitong},
  journal={arXiv preprint arXiv:2011.07553},
  year={2020}
}
```

import os
import optuna
import traceback
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from optuna_dashboard import run_server
from huggingface_sb3 import package_to_hub

class EvalSaveCallback(BaseCallback):
    ''' callback function used to evaluate and save the best model
    '''
    def __init__(self, eval_env, n_eval, score_threshold, save_path, trial_number=0):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval = n_eval
        self.score_threshold = score_threshold 
        self.save_path = save_path
        self.trial_number = trial_number

    def _on_step(self) -> bool:
        if self.n_calls % self.n_eval == 0:
            self.eval_env.reset()
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
            score = mean_reward - std_reward
            print(f'test at step: {self.n_calls}, score: {score}')
            if score >= self.score_threshold:
                path = os.path.join(self.save_path, f"trial_{self.trial_number}_step_{self.n_calls}_{score}.zip")  
                self.model.save(path)
                print(f"Save new best model with score {mean_reward:.2f} at trial {self.trial_number} step {self.n_calls}")
        return True

def optimize_params(trial, env_num=None):
    ''' generate a dict of params(for model & optuna)
    '''
    params = {
        # 'n_steps': trial.suggest_int('n_steps', 2000, 2500),
        'n_epochs': trial.suggest_int('n_epochs', 10, 20),
        'gamma': trial.suggest_float('gamma', 0.98, 1),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.95, 1)
    }
    # if env_num:
    #     params['batch_size'] = params['n_steps'] * env_num
    return params

env_id = 'LunarLander-v2'
env_num = 4
def objective(trial):
    ''' optuna function
    '''
    try:
        # time_steps = trial.suggest_int('time_steps', 1e+4, 1e+6)
        train_env = make_vec_env(env_id, n_envs=env_num)
        eval_env = make_vec_env(env_id, n_envs=env_num)
        params = optimize_params(trial, env_num=env_num)
        print(trial.number, params)
        model = PPO('MlpPolicy', train_env, verbose=1, **params)
        save_call_back = EvalSaveCallback(eval_env, 4096, 270, '../../models', trial_number=trial.number)

        model.learn(1e+6, callback=save_call_back, progress_bar=True)

        eval_env.reset()
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

        print(mean_reward, std_reward, mean_reward - std_reward)

        score = mean_reward - std_reward
        return score     


    except:
        print(traceback.format_exc())
        return -1
    

def upload_ppo_hug(model):
    ''' upload ppo to hugging face
    '''
    env_id = "LunarLander-v2"
    model_architecture = "PPO"
    repo_id = "zdmcode/ppo-LunarLander-v2"
    commit_message = "Upload PPO LunarLander-v2 trained agent"
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

    package_to_hub(
        model=model,  # Our trained model
        model_name='ppo_first_agent',  # The name of our trained model
        model_architecture=model_architecture,  # The model architecture we used: in our case PPO
        env_id=env_id,  # Name of the environment
        eval_env=eval_env,  # Evaluation Environment
        repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
        commit_message=commit_message,
    )


if __name__ == '__main__':
    storage = 'sqlite:///db.sqlite3'
    study = optuna.create_study(
        storage=storage,
        study_name="lunarlander_with_save_callback_2",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=20)

    run_server(storage)
    print(study.best_params)

    # best_model_path = r'models/trial_6_step_205000_281.89743073683985.zip'
    # model = PPO.load(best_model_path)
    # upload_ppo_hug(model)
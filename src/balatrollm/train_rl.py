import os
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from balatrollm.balatro_env import BalatroEnv

def mask_fn(env: BalatroEnv):
    # Returns the action mask
    return env.action_masks()

def main():
    parser = argparse.ArgumentParser(description="Train Balatro Deep RL agent")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps to train")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Balatro bot host")
    parser.add_argument("--port", type=int, default=12346, help="Balatro bot port")
    parser.add_argument("--seed", type=str, default="AAAAAAA", help="Game seed")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--logs-dir", type=str, default="logs/tensorboard", help="TensorBoard logs dir")
    
    import asyncio
    from balatrobot import BalatroInstance
    from balatrobot import Config as BalatrobotConfig

    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting Balatro Instance...")
    cfg = BalatrobotConfig.from_env()
    instance = BalatroInstance(cfg, port=args.port)
    asyncio.run(instance.start())
    
    # 1. Create Environment
    print("Initializing Balatro Environment...")
    env = BalatroEnv(host=args.host, port=args.port, seed=args.seed)
    
    # 2. Wrap with ActionMasker
    env = ActionMasker(env, mask_fn)
    
    # 3. Initialize Model
    # We use a relatively small neural network since the observation space is small (81 features)
    policy_kwargs = dict(net_arch=[128, 128])
    
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.logs_dir
    )
    
    # 4. Train
    print(f"Starting training for {args.timesteps} timesteps...")
    try:
        model.learn(total_timesteps=args.timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save model
        save_path = models_dir / "balatro_ppo_model"
        model.save(str(save_path))
        print(f"Model saved to {save_path}.zip")
        asyncio.run(instance.stop())
        print("Stopped Balatro Instance.")

if __name__ == "__main__":
    main()

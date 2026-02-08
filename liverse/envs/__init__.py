import os

os.environ["MUJOCO_GL"] = "egl"


def make_env(config, mode, id, liv=None):
    """Create environment based on configuration."""
    from liverse.envs import wrappers

    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        from liverse.envs.dmc import DeepMindControl
        env = DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "ML1":
        from liverse.envs.metaworld import MetaWorldEnvWrapper
        env = MetaWorldEnvWrapper(task_name=task, liv=liv, seed=config.seed + id, mode=mode)
    elif suite == "atari":
        from liverse.envs.atari import Atari
        env = Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        from liverse.envs.dmlab import DeepMindLabyrinth
        env = DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from liverse.envs.memorymaze import MemoryMaze
        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        from liverse.envs.crafter import Crafter
        env = Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        from liverse.envs.minecraft import make_env as make_minecraft_env
        env = make_minecraft_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

from gymnasium.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gridworld.env:GridWorldEnv',
)

register(
    id='Maze-v0',
    entry_point='gridworld.env:MazeEnv',
)

register(
    id='Cliff-v0',
    entry_point='gridworld.env:CliffEnv',
)

register(
    id='Ship-v0',
    entry_point='gridworld.env:ShipEnv',
)

register(
    id='ShipExploration-v0',
    entry_point='gridworld.env:ShipExplorationEnv',
)
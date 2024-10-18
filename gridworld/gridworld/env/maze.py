import numpy as np
import pygame

import  gymnasium as        gym
from    gymnasium import    spaces

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None, size=8) -> None:
        super(MazeEnv, self).__init__()
        self.size = size
        self.window_size = 512

        # Definizione del labirinto (0 = cella libera, 1 = muro)
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])

        # Posizione start, goal
        self.start = np.array([2, 0])
        self.goal = np.array([6, 7])

        # Spazio delle osservazioni:
        #  - Posizione agente
        #  - Posizione target
        self.observation_space = spaces.Dict(
            { 
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        
        # Definisco lo spazio delle azioni, saranno 4 possibili azioni NSWE 
        self.action_space = spaces.Discrete(4)

        # Conversione da azione discreta a movimento
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }   

        # Imposto la render mode, se passata correttamente altrimenti passo un errore
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # Funzione per ottenere le osservazioni 
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def reset(self, seed=None, options=None):
        # super() va a richiamare un attributo della classe genitore gym.Env
        # In questo caso serve ad assegnare un eventuale seed casuale
        super().reset(seed=seed)
        self._agent_location = self.start
        self._target_location = self.goal
        observation = self._get_obs()    
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):

        self._agent_location = self._update_location(action)
        
        # Check raggiungimento target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = - 1
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _update_location(self, action):
        direction = self._action_to_direction[action]
        new_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        if self.maze[tuple(new_location)] == 0:
            
            return new_location
        else:
            return self._agent_location
        

    # Funzione di rendering basata su Pygame
    def render(self):
        if self.render_mode== "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Genero un buffer su cui disegnare l'ambiente prima di disengarlo sullo schermo
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size/self.size)

        # Maze
        for i in range(self.size):
            for j in range(self.size):
                if self.maze[i, j] == 1:  
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),  
                        pygame.Rect(pix_square_size * j, pix_square_size * i, pix_square_size, pix_square_size)
                    )
        # Target
        pixel_position = self._target_location * pix_square_size
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (pixel_position[1], pixel_position[0], pix_square_size, pix_square_size),
        )

        # # Agente      
        pixel_position = self._agent_location * pix_square_size
        circle_center = (pixel_position[1] + pix_square_size // 2, pixel_position[0] + pix_square_size // 2)        
        
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            circle_center,
            pix_square_size / 3,
        )

        # Disegno la griglia
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # Copio il buffer appena creato su una finestra visibile
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            #  Mi assicuro di mantenere il framerate
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))

    # Funzione per chiudere la finestra di rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

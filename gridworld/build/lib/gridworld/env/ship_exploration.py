import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from gridworld.env import Unicycle

class ShipExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30} 

    def __init__(self, workspace=(0, 0, 10, 10), render_mode=None):
        """
        Inizializza l'ambiente di esplorazione della nave.

        Args:
            workspace (tuple): Limiti del workspace definiti come (min_x, min_y, max_x, max_y).
            render_mode (str, opzionale): Modalità di rendering, può essere 'human' o 'rgb_array'.

        Crea un'istanza della classe Unicycle, definisce gli spazi delle osservazioni e delle azioni,
        imposta la modalità di rendering e inizializza i dati per il rendering e la gridmap.
        """
        super(ShipExplorationEnv, self).__init__()
        self.workspace = workspace
        self.min_x = workspace[0]
        self.min_y = workspace[1]
        self.max_x = workspace[2]
        self.max_y = workspace[3]

        # Creo un'istanza della classe Unicycle
        self.agent = Unicycle(init_pose=[1, 1, 0], footprint={'radius': 0.1})
        self.robot_radius = self.agent.footprint['radius']
        self.dt = 1.0 / self.metadata['render_fps']

        # Definizione spazio delle osservazioni e delle azioni
        self.action_space = spaces.Discrete(3)
        self.action_to_controls = {
            0: 0.0, 
            1: self.agent.max_omega, 
            2: -self.agent.max_omega,
        }

        # Definizione spazio delle osservazioni
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Imposto la render mode, se passata correttamente altrimenti passo un errore
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Dati rendering
        self.window_size = (1600, 800)
        self.scale = self.window_size[1] / (self.max_y - self.min_y)
        self.window = None
        self.clock = None

        # Dati gridmap
        self.grid_resolution = 0.1
        self.grid_width = int((self.max_x - self.min_x) / self.grid_resolution)
        self.grid_height = int((self.max_y - self.min_y) / self.grid_resolution)
        self.gridmap = np.zeros((self.grid_height, self.grid_width))

    def get_obs(self):
        """
        Ottiene lo stato attuale del robot e lo normalizza per restituire un'osservazione.

        Il metodo ottiene le coordinate x, y e l'orientamento theta del robot. 
        Questi valori vengono poi normalizzati per essere compresi nell'intervallo [-1, 1].
        La normalizzazione di x e y viene effettuata rispetto ai limiti del workspace, 
        mentre theta viene normalizzato rispetto a [-pi, pi].

        Returns:
            observation (np.array): Un array contenente le coordinate normalizzate x, y e theta del robot.
        """
        # Ottieni lo stato del robot
        x, y, theta = self.agent.get_state()[:3] 
        # Normalizzazione di x e y dall'intervallo [min_x, max_x] a [-1, 1]
        x_norm = 2 * (x - self.min_x) / (self.max_x - self.min_x) - 1
        y_norm = 2 * (y - self.min_y) / (self.max_y - self.min_y) - 1

        # Normalizzazione di theta dall'intervallo [-pi, pi] a [-1, 1]
        theta_norm = theta / np.pi

        # Combina le osservazioni normalizzate
        observation = np.array([x_norm, y_norm, theta_norm], dtype=np.float32)
        return observation

    def get_info(self):
        """
        Restituisce informazioni aggiuntive sull'ambiente.

        Questo metodo restituisce informazioni aggiuntive sull'ambiente, 
        come il workspace, la posizione del robot e i limiti del workspace.

        Returns:
            info (dict): Un dizionario contenente informazioni aggiuntive sull'ambiente.
        """
        info = {
            "workspace": self.workspace,
            "robot_position": self.agent.get_state()[:2],
            "workspace_limits": (self.min_x, self.min_y, self.max_x, self.max_y)
        }	
        return info
    
    def reset(self, seed: int = None, options=None):
        """
        Resetta l'ambiente all'inizio di un nuovo episodio.

        Questo metodo resetta lo stato del robot e restituisce l'osservazione iniziale.
        Se la modalità di rendering è impostata su 'human', viene renderizzato il primo frame.

        Args:
            seed (int, opzionale): Il seme per il generatore di numeri casuali.
            options (dict, opzionale): Opzioni aggiuntive per il reset.

        Returns:
            observation (np.array): L'osservazione iniziale dell'ambiente.
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        super().reset(seed=seed, options=options)
        self.agent.reset()
        observation = self.get_obs()
        info = self.get_info()

        if self.render_mode == 'human':
            self.render_frame()
        return observation, info

    def step(self, action):
        """
        Esegue un passo nell'ambiente in base all'azione fornita.

        Questo metodo aggiorna lo stato del robot in base all'azione scelta, 
        calcola la nuova osservazione, la ricompensa, e verifica se l'episodio è terminato.
        Se la modalità di rendering è impostata su 'human', viene renderizzato un frame.

        Args:
            action (int): L'azione scelta dall'agente.

        Returns:
            observation (np.array): La nuova osservazione dell'ambiente.
            reward (float): La ricompensa ottenuta dopo aver eseguito l'azione.
            terminated (bool): Indica se l'episodio è terminato.
            truncated (bool): Indica se l'episodio è stato troncato.
            info (dict): Informazioni aggiuntive sull'ambiente.
        """
        omega = self.action_to_controls[action]
        self.agent.simple_kinematics(omega=omega, dt=self.dt)
        
        observation = self.get_obs()
        reward = 0
        terminated = not self.agent.boundary_check(self.workspace)
        truncated = False
        info = self.get_info()

        if self.render_mode == 'human':
            self.render_frame()

        return observation, reward, terminated, truncated, info

    def pygame_init(self) -> None:
        """
        Inizializza pygame per il rendering.

        Questo metodo inizializza la finestra di pygame e il clock se non sono già stati inizializzati.
        Viene chiamato all'inizio del rendering per assicurarsi che pygame sia pronto.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.surface_width = self.window_size[0] // 2
            self.surface_height = self.window_size[1]
            self.real_world_canvas = pygame.Surface((self.surface_width, self.surface_height))
            self.gridmap_canvas = pygame.Surface((self.surface_width, self.surface_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def render_real_world(self):
        """
        Renderizza il mondo reale.

        Questo metodo disegna il robot e la sua direzione sulla canvas del mondo reale.
        Le coordinate del robot vengono convertite in pixel in base alla scala della finestra.
        """
        self.real_world_canvas.fill((255, 255, 255))  # Sfondo bianco

        # Posizione del robot in pixel
        robot_pixel_pos = (
            int((self.agent.x - self.min_x) * self.scale),
            int(self.surface_height - (self.agent.y - self.min_y) * self.scale)
        )

        # Raggio del robot in pixel
        robot_radius_pixels = int(self.robot_radius * self.scale)

        # Disegna il robot
        pygame.draw.circle(
            self.real_world_canvas,
            (0, 0, 255),
            robot_pixel_pos,
            robot_radius_pixels
        )

        # Disegna la direzione del robot
        heading_length = robot_radius_pixels * 2
        heading_end_pos = (
            robot_pixel_pos[0] + int(heading_length * np.cos(self.agent.theta)),
            robot_pixel_pos[1] - int(heading_length * np.sin(self.agent.theta))
        )
        pygame.draw.line(
            self.real_world_canvas,
            (255, 0, 0),
            robot_pixel_pos,
            heading_end_pos,
            2
        )

    def render_frame(self):
            """
            Renderizza un frame dell'ambiente.

            Questo metodo chiama pygame_init per assicurarsi che pygame sia inizializzato,
            quindi renderizza il mondo reale e la gridmap (se implementata).
            Infine, aggiorna la finestra di pygame con i nuovi disegni.
            """
            self.pygame_init()
            self.render_real_world()
            # self.render_gridmap()

            self.window.blit(self.real_world_canvas, (0, 0))
            self.window.blit(self.gridmap_canvas, (self.surface_width, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        """
        Chiude l'ambiente e pygame.

        Questo metodo chiude la finestra di pygame e termina il modulo pygame.
        Viene chiamato quando l'ambiente non è più necessario.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    """" DEBUG """
    env = ShipExplorationEnv(render_mode='human')
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        print(observation)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
import numpy as np
import warnings

class Unicycle():
    def __init__(self, init_pose=[0,0,0], footprint={}):
        """ Costruttore per la classe Unicycle """

        # Inizializza lo stato del monociclo
        self.init_pose = init_pose
        self.reset()

        # Definisce le velocità massime lineari e angolari
        self.max_v      = 1
        self.max_omega  = 2.5

        # Definisce l'impronta del monociclo
        if not isinstance(footprint, dict):
            warnings.warn("Impronta non definita. Imposto l'impronta a punto.")
            footprint = {}
        elif 'radius' not in footprint and 'square' not in footprint:
            warnings.warn("Impronta non definita. Imposto l'impronta a punto.")
            footprint = {}
        self.footprint = footprint

        # Definisce i parametri del sensore LIDAR
        self.lidar_params = {'FoV': np.deg2rad(60), 'max_range': 3.0, 'n_beams': 60}

    def reset(self):
        """
        Resetta lo stato del monociclo.
        Questo metodo imposta la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega) del monociclo
        ai loro valori iniziali.
        """
        self.x      = self.init_pose[0]
        self.y      = self.init_pose[1]
        self.theta  = self.init_pose[2]
        self.v      = 0
        self.omega  = 0

    def incremental_kinematics(self, dv=0.0, domega=0.0, dt=1.0):
        """
        Aggiorna lo stato del modello di monociclo in base ai cambiamenti forniti di velocità e velocità angolare.
        Parametri:
            dv (float): Cambiamento nella velocità lineare.
            domega (float): Cambiamento nella velocità angolare.
        Questa funzione modifica la velocità attuale e la velocità angolare del monociclo, assicurandosi che rimangano entro i limiti specificati. 
        Successivamente aggiorna la posizione (x, y) e l'orientamento (theta) del monociclo in base alla nuova velocità e velocità angolare.
        """
        # Aggiorna la velocità e la velocità angolare
        self.v      += dv
        self.omega  += domega
        # Limita la velocità e la velocità angolare per rimanere entro i limiti specificati
        self.v      = np.clip(self.v, -self.max_v, self.max_v)
        self.omega  = np.clip(self.omega, -self.max_omega, self.max_omega)
        # Aggiorna la posizione e l'orientamento del monociclo
        self.x      += self.v * np.cos(self.theta) * dt
        self.y      += self.v * np.sin(self.theta) * dt
        self.theta  = self.wrapToPi(self.theta + self.omega * dt)
        
    def simple_kinematics(self, omega, dt=1.0):
        """
        Aggiorna la posizione del robot monociclo utilizzando la cinematica semplice.
        Parametri:
        - omega (float): Velocità angolare del robot.
        - dt (float): Intervallo di tempo per l'aggiornamento.
        Ritorna:
        Nessuno
        """
        self.omega = omega
        self.v = self.max_v
        self.theta = self.wrapToPi(self.theta + self.omega * dt)
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

    def wrapToPi(self, angle):
        """
        Converte l'angolo di input nell'intervallo [-pi, pi].
        Parametri:
            angle (float): L'angolo di input da convertire.
        Ritorna:
            float: L'angolo convertito nell'intervallo [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def get_state(self):
        """
        Restituisce lo stato attuale del monociclo.
        Ritorna:
            np.array: Un array contenente la posizione (x, y), l'orientamento (theta), la velocità lineare (v) e la velocità angolare (omega).
        """
        return np.array([self.x, self.y, self.theta, self.v, self.omega])
    
    def collision_check(self, obstacles):
        """
        Controlla le collisioni tra l'impronta dell'oggetto e una lista di ostacoli.
        Questa funzione itera attraverso una lista di ostacoli e verifica se qualche punto 
        dell'impronta dell'oggetto interseca con il rettangolo di delimitazione degli ostacoli. 
        La posizione e l'orientamento dell'oggetto vengono presi in considerazione per calcolare le 
        coordinate effettive dei punti dell'impronta.
        Parametri:
            obstacles (list): Una lista di ostacoli, dove ogni ostacolo è definito 
                              da una tupla di quattro valori (xmin, ymin, xmax, ymax) 
                              che rappresentano il rettangolo di delimitazione.
        Ritorna:
            bool: True se viene rilevata una collisione, False altrimenti.
        """
        # Impronta circolare
        if 'radius' in self.footprint:  
            radius = self.footprint['radius']
            for obstacle in obstacles:
                # Controlla se la distanza dal centro all'ostacolo è inferiore al raggio
                if (self.x - (obstacle[0] + obstacle[2]) / 2) ** 2 + (self.y - (obstacle[1] + obstacle[3]) / 2) ** 2 < radius ** 2:
                    return True

        # Impronta quadrata       
        elif 'square' in self.footprint:  
            for obstacle in obstacles:
                for point in self.footprint['square']:
                    x = self.x + point[0] * np.cos(self.theta) - point[1] * np.sin(self.theta)
                    y = self.y + point[0] * np.sin(self.theta) + point[1] * np.cos(self.theta)
                    if obstacle[0] < x < obstacle[2] and obstacle[1] < y < obstacle[3]:
                        return True
        # Nessuna impronta definita 
        else:
            for obstacle in obstacles:
                if obstacle[0] < self.x < obstacle[2] and obstacle[1] < self.y < obstacle[3]:
                    return True
        return False
    
    def boundary_check(self, workspace):
        """
        Controlla se tutti i punti dell'impronta dell'oggetto sono all'interno del workspace specificato.
        Questa funzione itera attraverso i punti dell'impronta e verifica se ciascun punto 
        è all'interno del rettangolo di delimitazione definito dal workspace.
        Parametri:
            workspace (tuple): Una tupla di quattro valori (xmin, ymin, xmax, ymax) 
                               che rappresentano il rettangolo di delimitazione del workspace.
        Ritorna:
            bool: True se tutti i punti sono all'interno del workspace, False altrimenti.
        """
        # Impronta circolare
        if 'radius' in self.footprint:
            if (workspace[0] + self.footprint['radius'] <= self.x <= workspace[2] - self.footprint['radius'] and
                workspace[1] + self.footprint['radius'] <= self.y <= workspace[3] - self.footprint['radius']):
                return True
            return False
            
        # Impronta quadrata
        elif 'square' in self.footprint: 
            for point in self.footprint['square']:
                x = self.x + point[0] * np.cos(self.theta) - point[1] * np.sin(self.theta)
                y = self.y + point[0] * np.sin(self.theta) + point[1] * np.cos(self.theta)
                if not (workspace[0] <= x <= workspace[2] and workspace[1] <= y <= workspace[3]):
                    return False
            return True
        
        # Nessuna impronta definita
        else:
            if workspace[0] <= self.x <= workspace[2] and workspace[1] <= self.y <= workspace[3]:
                return True
            return False

    def lidar(self, obstacles):
        """
        Calcola le misurazioni di distanza dal sensore Lidar.
        Parametri:
            obstacles (list): Lista di ostacoli nell'ambiente.
        Ritorna:
            numpy.ndarray: Array di misurazioni di distanza dal sensore Lidar.
        """
        # Inizializza le distanze al valore massimo
        ranges = np.full(self.lidar_params['n_beams'], self.lidar_params['max_range'])
        FoV = self.lidar_params['FoV']
        n_beams = self.lidar_params['n_beams']

        # Angoli assoluti dei raggi
        angles = np.linspace(- FoV / 2, FoV / 2, n_beams) + self.theta

        for i, angle in enumerate(angles):
            min_distance = self.lidar_params['max_range']
            for obstacle in obstacles:
                distance = self.ray_rectangle_intersection(self.x, self.y, angle, obstacle)
                if distance is not None and distance < min_distance:
                    min_distance = distance
            ranges[i] = min_distance

        return ranges
    
    def ray_rectangle_intersection(self, x0, y0, angle, obstacle):
        """
        Calcola la distanza minima tra un raggio e un ostacolo rettangolare.
        Parametri:
        - x0 (float): La coordinata x dell'origine del raggio.
        - y0 (float): La coordinata y dell'origine del raggio.
        - angle (float): L'angolo del raggio in radianti.
        - obstacle (tuple): Una tupla contenente le coordinate dell'ostacolo rettangolare nel formato (x_min, y_min, x_max, y_max).
        Ritorna:
        - min_distance (float): La distanza minima tra il raggio e l'ostacolo rettangolare.
        Nota:
        - Il raggio è definito dalla sua origine (x0, y0) e dal suo angolo.
        - L'ostacolo è definito dalle sue coordinate minime e massime x e y.
        - La funzione calcola il punto di intersezione tra il raggio e ciascun lato dell'ostacolo rettangolare.
        - La distanza minima è la distanza più breve tra l'origine del raggio e qualsiasi punto di intersezione.
        """
        # Estrae le coordinate dell'ostacolo
        x_min, y_min, x_max, y_max = obstacle
        # Calcola i 4 lati dell'ostacolo (segmenti)
        edges = [((x_min, y_min), (x_max, y_min)),
                 ((x_max, y_min), (x_max, y_max)),
                 ((x_max, y_max), (x_min, y_max)),
                 ((x_min, y_max), (x_min, y_min))]
        # Inizializza la distanza minima a None
        min_distance = None
        # Direzione del raggio
        dx = np.cos(angle)
        dy = np.sin(angle)
        # Itera sui lati dell'ostacolo
        for edge in edges:
            # Calcola il punto di intersezione tra il raggio e il lato
            point = self.ray_segment_intersection(x0, y0, dx, dy, edge)
            if point is not None:
                # Calcola la distanza tra il punto di intersezione e l'origine
                distance = np.hypot(point[0] - x0, point[1] - y0)
                if min_distance is None or distance < min_distance:
                    min_distance = distance
        return min_distance
    
    def ray_segment_intersection(self, x0, y0, dx, dy, segment):
        """
        Calcola il punto di intersezione tra un raggio e un segmento di linea.
        Parametri:
        - x0 (float): Coordinata x del punto di partenza del raggio.
        - y0 (float): Coordinata y del punto di partenza del raggio.
        - dx (float): Componente x del vettore direzione del raggio.
        - dy (float): Componente y del vettore direzione del raggio.
        - segment (tuple): Tupla contenente le coordinate degli estremi del segmento di linea nel formato ((x1, y1), (x2, y2)).
        Ritorna:
        - tuple o None: Se esiste un punto di intersezione, restituisce una tupla (ix, iy) contenente le coordinate x e y del punto di intersezione. Se non esiste un punto di intersezione, restituisce None.
        """
        # Estrae le coordinate del segmento
        (x1, y1), (x2, y2) = segment
        x3, y3 = x0, y0
        x4, y4 = x0 + dx, y0 + dy
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None # Le linee sono parallele o sovrapposte
        
        # Calcola i parametri di intersezione
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if 0 <= t <= 1 and u >= 0:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        else:
            return None
                
if __name__ == "__main__":
    # footprint = {'square': [[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]]}
    footprint = {'radius': 0.05}
    # Crea un'istanza della classe Unicycle
    unicycle = Unicycle(init_pose=[0, 2, np.pi/6], footprint=0)
    
    # Esempio di aggiornamento dello stato
    # unicycle.simple_kinematics(omega=10)
    
    # Ottieni lo stato attuale
    state = unicycle.get_state()
    print("Stato attuale:", state)
    
    # Definisci alcuni ostacoli
    obstacles = [(1, 1, 2, 2), (3, 3, 4, 4)]
    
    # Controlla le collisioni
    collision = unicycle.collision_check(obstacles)
    print("Collisione rilevata:", collision)
    
    # Definisci i limiti del workspace
    workspace = (0, 0, 5, 5)
    
    # Controlla se è all'interno dei limiti
    within_bounds = unicycle.boundary_check(workspace)
    print("All'interno dei limiti del workspace:", within_bounds)

    # ranges, angles = unicycle.lidar(obstacles)
    # print("Distanze:", ranges)
    # print("Angoli:", angles)

    ranges = unicycle.lidar(obstacles)
    print(ranges)
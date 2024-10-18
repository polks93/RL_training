from torch.distributions.normal import Normal

import numpy as np
import torch
import torch.nn as nn

""" Definisco una classe del tipo nn contenente la policy che l'agente dovrà migliorare
    durante la fase di RL tramite l'algoritmo REINFORCE """
class Policy_Network(nn.Module):

    """ Init neural network che stima media e deviazione standard della distribuzione
        normale delle azioni
        Input:  - Dimensione dello spazio di osservazione
                - Dimensione dello spazio delle azioni """
    
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        
        super().__init__()
        
        hidden_space1 = 16
        hidden_space2 = 32

        #  Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Media e deviazione standard della policy network
        self.policy_mean_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))
        self.policy_stddev_net = nn.Sequential(nn.Linear(hidden_space2, action_space_dims))

    """ Funzione che data un'osservazione x dell'env, fornisce una predizione
        della media e della deviazione standard della distribuzione normale delle azioni """
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        shared_features = self.shared_net(x.float())
        
        action_means = self.policy_mean_net(shared_features)
        action_stddev = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means, action_stddev

class REINFORCE:
    def __init__(self, obs_space_dims: int, action_space_dims: int):

        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.eps = 1e-6

        self.probs = []
        self.rewards = []

        # Rete neurale che decide la policy
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    """ Funzione che sceglie un azione basandosi sulla policy e sull'osservazione 
        Input: Osservazione dell'env
        Output: Azione da eseguire """
    def sample_action(self, state: np.ndarray) -> float:
        
        state = torch.tensor(np.array([state]))

        # Ottengo media e std dev della distribuzione di probabilità delle azioni
        action_means, action_stddevs = self.net(state)

        # Genero una distribuzione normale 
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)

        #  Campiono l'azione dalla distribuzione normale con la sua probabilitá 
        action = distrib.sample()
        prob = distrib.log_prob(action)

        #  Conversione azione da torch.Tensor a numpy.ndarray
        action = action.numpy()

        # Salvo la probabilitá associata
        self.probs.append(prob)

        return action

    """ Update dei pesi della policy network in base ai risultati ottenuti durante un
        episodio di interazone con l'ambiente """
    def update(self):

        # Init varaibile che tiene traccia del return cumulativo 
        running_g = 0
        #  Lista che conterrá il return scontato calcolato in ogni stato
        gs = []

        # Scorro self.rewards a ritroso
        for R in self.rewards[::-1]:
            # Calculo return Gt = Rt + gamma*G(t+1)
            running_g = R + self.gamma * running_g
            # Salvo Gt all'inizio della lista gs. In questo modo hanno la stessa sequenza temporale degli stati
            gs.insert(0, running_g)
    
        # Converto tutti i ritorni scontati in un tensore
        deltas = torch.tensor(gs)
    
        # Calcolo della perdita loss in base a gs
        loss = 0

        for log_prob, delta in zip(self.probs, deltas): # zip() scorre contemporaneamente sui due argomenti 
            # Per ogni coppia calcolo un contributo alla perdita
            # Si usa il -1 per minimizzare la perdita negativa
            loss += log_prob.mean() * delta * (-1)
    
        #  Update pesi della policy
        self.optimizer.zero_grad()   # Azzero i gradienti accumulati dall'iterazione precedente
        loss.backward()              # calcolo i nuovi gradienti della perdita con la policy corrente (backward propagation)
        self.optimizer.step()        # Aggiorno i pesi della rete usando i gradienti calcolati
    
        # Azzero i vettori alla fine dell'episodio
        self.probs = []
        self.rewards = []
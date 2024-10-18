Idee shipexploration v0:
- reward sparse raccoglibili
- zona quadrata come stato terminale
- ostacolo centrale
- stato composto da x y theta + n_reward raccolte eventualmente
- no lidar 
variante:
- stato composto da heading e distanza ostacolo + vicino
- no pose nello stato
- ostacoli random

Idee shipexploration v1:
- nello stato dell'agente formato da 2 componenti:
    - heading e distanza cluster di celle di frontiera
    - heading e distanza cluster di celle di contorno
- reward basato su celle di frontiera e di contorno
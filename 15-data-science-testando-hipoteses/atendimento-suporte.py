# %%
import numpy as np
from scipy.stats import ttest_1samp

# %%# Tempo de resposta em minutos para 25 solicitações de suporte
tempo_resposta = [
    28,
    32,
    29,
    31,
    30,
    33,
    28,
    30,
    31,
    29,
    30,
    32,
    29,
    31,
    30,
    33,
    28,
    30,
    31,
    29,
    30,
    32,
    24,
    29,
    30,
]

# %%
## Media do tempo
np.mean(tempo_resposta)

# %%
# Realiznado teste unilateral teste T
hipotese_nula = 30
tipo_hipotese_alternativa = (
    "less"  # teste unilateral para verificar se a média é menor que 30 minutos
)
estatica_teste, p_valor = ttest_1samp(
    tempo_resposta, popmean=hipotese_nula, alternative=tipo_hipotese_alternativa
)

# %%
# Interpretação do resultado
nivel_significancia = 0.05

if p_valor < nivel_significancia:
    print("Rejeitamos a hipótese nula: O tempo médio de resposta é menor que 30 minutos.")
else:
    print("Não rejeitamos a hipótese nula: Não há evidências suficientes para afirmar que o tempo médio de resposta é menor que 30 minutos.") 
    
# %%

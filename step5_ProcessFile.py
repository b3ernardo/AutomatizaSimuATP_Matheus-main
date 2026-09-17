import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from f_readpl4 import readpl4
from Functions.metrics import normalize_zscore
from scipy.stats import entropy, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings(action="ignore"):
    fxn()

def listar_arquivos_pl4(caminho_pasta: str, recursivo: bool = True) -> list[str]:
    pasta = Path(caminho_pasta)

    if not pasta.is_dir():
        raise ValueError(f"O diretório '{caminho_pasta}' não existe.")

    # rglob busca recursivamente (inclui subpastas); glob busca apenas na raiz
    padrao = "**/*.pl4" if recursivo else "*.pl4"

    return [str(arquivo.resolve()) for arquivo in pasta.glob(padrao)]

pasta_alvo = str(Path.cwd()) + r"\\step2_ArquivosPL4"
arquivos_pl4 = listar_arquivos_pl4(pasta_alvo, recursivo=True)

print(f"Total de arquivos encontrado: {len(arquivos_pl4)}")
for arq in arquivos_pl4:
    print(arq)

df = []
y = np.array([])

for item in arquivos_pl4:
    pl4 = readpl4(item)
    for i, var_info in enumerate(pl4['var_info']):
        if 'SUBNEU' in var_info.get('FROM', '') or 'SUBNEU' in var_info.get('TO', ''):
        # if 'GER' in var_info.get('FROM', ''):
            df.append(pl4['data'][i])
            if 'Evento' in item: # já insere a label correta para y
                y = np.append(y, int(0))  # Evento
            else:
                y = np.append(y, int(1))  # FAI
            # df.append(pl4['data'][121])
            # df.append(pl4['data'][122])
            # df.append(pl4['data'][123])

df = pd.DataFrame(df)
df_normalized = normalize_zscore(df.T)
w = [256]*len(df_normalized.columns)  # Assuming you want to apply the same window size for all columns
fs = [256*60]*len(df_normalized.columns)
f_system = 60
df_norm_inicio = df_normalized.iloc[15360:,:].reset_index(drop=True)

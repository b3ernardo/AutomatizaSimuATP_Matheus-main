import numpy as np
import pandas as pd

def extrair_harmonicas(sinal, amostras_por_ciclo=64, ordem_maxima=20):
    """
    Extrai as componentes harmônicas de um sinal, ciclo a ciclo.

    Parâmetros
    ----------
    sinal : array-like
        Amostras do sinal no domínio do tempo.

    amostras_por_ciclo : int
        Número de amostras existentes em cada ciclo.

    ordem_maxima : int
        Maior ordem harmônica a ser extraída.

    Retorna
    -------
    pd.DataFrame
        DataFrame em que cada linha representa um ciclo e cada
        coluna representa uma ordem harmônica.

        Os valores são complexos, preservando módulo e fase.
    """

    sinal = np.asarray(sinal)

    # ========================================================
    # Validações
    # ========================================================
    if len(sinal) % amostras_por_ciclo != 0:
        raise ValueError(
            "O número de amostras do sinal deve ser múltiplo "
            f"de {amostras_por_ciclo}."
        )

    limite_nyquist = amostras_por_ciclo // 2

    if ordem_maxima > limite_nyquist:
        raise ValueError(
            f"A ordem máxima permitida é {limite_nyquist} "
            f"para {amostras_por_ciclo} amostras por ciclo."
        )

    # ========================================================
    # Divisão do sinal em ciclos
    # ========================================================
    numero_ciclos = len(sinal) // amostras_por_ciclo

    ciclos = sinal.reshape(
        numero_ciclos,
        amostras_por_ciclo,
    )

    # ========================================================
    # FFT ciclo a ciclo
    # ========================================================
    fft_ciclos = np.fft.rfft(
        ciclos,
        axis=1,
    )

    # Normalização
    fft_ciclos = fft_ciclos / amostras_por_ciclo

    # ========================================================
    # Seleção das harmônicas H1 ... Hn
    # ========================================================
    harmonicas = fft_ciclos[
        :,
        1 : ordem_maxima + 1
    ]

    # ========================================================
    # DataFrame de saída
    # ========================================================
    colunas = [
        f"H{ordem_harmonica}"
        for ordem_harmonica in range(
            1,
            ordem_maxima + 1
        )
    ]

    return pd.DataFrame(
        harmonicas,
        columns=colunas,
    )
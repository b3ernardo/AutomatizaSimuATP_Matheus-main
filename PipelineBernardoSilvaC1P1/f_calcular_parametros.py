import numpy as np

def calcular_parametros(harmonicas_tensao, harmonicas_corrente):
    """
    Calcula os parâmetros associados às componentes harmônicas
    de tensão e corrente, ciclo a ciclo.

    Parâmetros
    ----------
    harmonicas_tensao : pd.DataFrame
        Componentes harmônicas complexas da tensão.
        Linhas = ciclos
        Colunas = H1, H2, ..., H20

    harmonicas_corrente : pd.DataFrame
        Componentes harmônicas complexas da corrente.

    Retorna
    -------
    dict
        Dicionário contendo os parâmetros calculados para
        cada ciclo e ordem harmônica.
    """

    V = harmonicas_tensao.to_numpy()
    I = harmonicas_corrente.to_numpy()

    # Validação
    if V.shape != I.shape:
        raise ValueError(
            "As matrizes de tensão e corrente devem possuir "
            "as mesmas dimensões."
        )

    # Módulos
    modulo_tensao = np.abs(V)
    modulo_corrente = np.abs(I)

    # Ângulos
    angulo_tensao = np.angle(V)
    angulo_corrente = np.angle(I)

    # Energia
    energia_tensao = modulo_tensao ** 2
    energia_corrente = modulo_corrente ** 2

    # Diferença angular V-I
    angulo_fator_potencia = np.angle(V * np.conj(I))

    # Potência ativa
    potencia_ativa = np.real(V * np.conj(I))

    # Impedância
    impedancia = np.divide(
        modulo_tensao,
        modulo_corrente,
        out=np.full_like(
            modulo_tensao,
            np.nan,
            dtype=float,
        ),
        where=modulo_corrente != 0,
    )

    return {
        "energia_tensao": energia_tensao,
        "modulo_tensao": modulo_tensao,
        "angulo_tensao": angulo_tensao,
        "energia_corrente": energia_corrente,
        "modulo_corrente": modulo_corrente,
        "angulo_corrente": angulo_corrente,
        "angulo_fator_potencia": angulo_fator_potencia,
        "impedancia": impedancia,
        "potencia_ativa": potencia_ativa,
    }
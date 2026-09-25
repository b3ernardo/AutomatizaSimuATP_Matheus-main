import numpy as np

def calcular_estatisticas(parametros):
    """
    Calcula estatísticas dos parâmetros ao longo dos ciclos.

    Para cada parâmetro e cada harmônica são calculados:
        - média
        - máximo
        - desvio padrão
        - variância
        - mínimo
        - amplitude

    Parâmetros
    ----------
    parametros : dict
        Dicionário retornado por calcular_parametros().
        Cada valor deve possuir formato:
        (numero_ciclos, numero_harmonicas)

    Retorna
    -------
    dict
        Dicionário contendo um atributo para cada combinação:
        parâmetro x harmônica x estatística.
    """

    atributos = {}

    for nome_parametro, valores in parametros.items():
        # ====================================================
        # Estatísticas ao longo dos ciclos
        # ====================================================
        media = np.nanmean(valores, axis=0)
        maximo = np.nanmax(valores, axis=0)
        desvio_padrao = np.nanstd(valores, axis=0)
        variancia = np.nanvar(valores, axis=0)
        minimo = np.nanmin(valores, axis=0)
        amplitude = maximo - minimo

        # ====================================================
        # Criação dos atributos
        # ====================================================
        numero_harmonicas = valores.shape[1]

        for indice in range(numero_harmonicas):
            harmonica = indice + 1
            prefixo = (f"{nome_parametro}_H{harmonica}")

            atributos[f"{prefixo}_media"] = media[indice]
            atributos[f"{prefixo}_maximo"] = maximo[indice]
            atributos[f"{prefixo}_desvio_padrao"] = desvio_padrao[indice]
            atributos[f"{prefixo}_variancia"] = variancia[indice]
            atributos[f"{prefixo}_minimo"] = minimo[indice]
            atributos[f"{prefixo}_amplitude"] = amplitude[indice]

    return atributos
def selecionar_sinais_ponto(df, ponto, incluir_time=True):
    """
    Seleciona os sinais de tensão e corrente associados
    a um ponto de medição.

    Parâmetros:
        df:
            DataFrame contendo os sinais da simulação.

        ponto:
            Ponto de medição que será selecionado.
            Exemplos:
                "SUB"
                "M02"
                "M03"
                "M09"

        incluir_time:
            Define se a coluna de tempo será mantida.

    Retorna:
        DataFrame contendo os sinais disponíveis no ponto
        selecionado e, opcionalmente, a coluna de tempo.
    """

    ponto = ponto.upper()

    # Seleção da SUB
    if ponto == "SUB":
        colunas_tensao = [
            coluna
            for coluna in df.columns
            if coluna.startswith("SUB")
            and "(V-node)" in coluna
        ]

        colunas_corrente = [
            coluna
            for coluna in df.columns
            if "-SUB" in coluna
            and "(I-bran)" in coluna
        ]
    # Seleção dos demais pontos de medição
    else:
        colunas_tensao = [
            coluna
            for coluna in df.columns
            if coluna.startswith(ponto)
            and "(V-node)" in coluna
        ]

        colunas_corrente = [
            coluna
            for coluna in df.columns
            if f"-{ponto}" in coluna
            and "(I-bran)" in coluna
        ]

    # Validação
    if not colunas_tensao:
        raise ValueError(
            f"Nenhum sinal de tensão encontrado para {ponto}."
        )

    if not colunas_corrente:
        raise ValueError(
            f"Nenhum sinal de corrente encontrado para {ponto}."
        )

    # Montagem das colunas finais
    colunas = (
        colunas_tensao
        + colunas_corrente
    )

    if incluir_time:
        colunas.insert(0, "time")

    return df[colunas].copy()
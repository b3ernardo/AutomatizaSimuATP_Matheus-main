FREQUENCIA_REDE = 60.0
AMOSTRAS_POR_CICLO = 64

def recortar_janela(df, ciclos_antes, ciclos_depois):
    """
    Recorta uma janela temporal do DataFrame em torno de TTOQ (t = 0).

    Parâmetros:
        df:
            DataFrame contendo a coluna 'time'.

        ciclos_antes:
            Quantidade de ciclos antes de TTOQ.

        ciclos_depois:
            Quantidade de ciclos depois de TTOQ.

    Retorna:
        DataFrame contendo apenas a janela solicitada.
    """

    # Conversão de ciclos para segundos
    tempo_antes = ciclos_antes / FREQUENCIA_REDE
    tempo_depois = ciclos_depois / FREQUENCIA_REDE

    # Recorte da janela
    df_recortado = df[
        (df["time"] >= -tempo_antes)
        & (df["time"] < tempo_depois)
    ].copy()

    # Validação da quantidade de amostras
    total_esperado = (
        (ciclos_antes + ciclos_depois)
        * AMOSTRAS_POR_CICLO
    )

    if len(df_recortado) != total_esperado:
        raise ValueError(
            f"Quantidade inesperada de amostras: "
            f"{len(df_recortado)}. "
            f"Esperado: {total_esperado}."
        )

    return df_recortado
def preparar_sinais(df_ponto, fase_faltosa):
    """
    Prepara as representações dos sinais utilizadas na
    extração de atributos.

    Para pontos trifásicos:
        - soma das três fases;
        - fase faltosa.

    Para pontos monofásicos:
        - utiliza o único sinal disponível.
    """

    fase_faltosa = fase_faltosa.upper()

    if fase_faltosa not in {"A", "B", "C"}:
        raise ValueError(
            f"Fase faltosa inválida: {fase_faltosa}"
        )

    # Identificação das tensões e correntes
    colunas_tensao = [
        coluna for coluna in df_ponto.columns
        if coluna.endswith("(V-node)")
    ]

    colunas_corrente = [
        coluna for coluna in df_ponto.columns
        if coluna.endswith("(I-bran)")
    ]

    # Caso trifásico
    if len(colunas_tensao) == 3 and len(colunas_corrente) == 3:
        # Soma das três fases
        tensao_soma = df_ponto[colunas_tensao].sum(axis=1)
        corrente_soma = df_ponto[colunas_corrente].sum(axis=1)

        # Identificação da fase faltosa
        coluna_tensao_fase = encontrar_coluna_fase(
            colunas_tensao,
            fase_faltosa,
        )

        coluna_corrente_fase = encontrar_coluna_fase(
            colunas_corrente,
            fase_faltosa,
        )

        return {
            "soma": {
                "tensao": tensao_soma,
                "corrente": corrente_soma,
            },
            "fase_faltosa": {
                "tensao": df_ponto[coluna_tensao_fase],
                "corrente": df_ponto[coluna_corrente_fase],
            },
        }

    # Caso monofásico
    if len(colunas_tensao) == 1 and len(colunas_corrente) == 1:
        return {
            "monofasico": {
                "tensao": df_ponto[colunas_tensao[0]],
                "corrente": df_ponto[colunas_corrente[0]],
            }
        }

    # Configuração inesperada
    raise ValueError(
        "Quantidade inesperada de sinais: "
        f"{len(colunas_tensao)} tensões e "
        f"{len(colunas_corrente)} correntes."
    )

def encontrar_coluna_fase(colunas, fase):
    """
    Identifica qual coluna corresponde à fase A, B ou C.

    Exemplos reconhecidos:
        SUBA (V-node)
        M02FA (V-node)
        GERA-SUBA (I-bran)
        N02FA-M02FA (I-bran)
    """

    for coluna in colunas:
        nome = coluna.split(" (")[0]

        if nome.endswith(f"SUB{fase}"): # SUBA / SUBB / SUBC
            return coluna

        if nome.endswith(f"F{fase}"): # M02FA / M02FB / M02FC
            return coluna

        # Correntes da SUB
        if nome.endswith(f"-SUB{fase}"): # GERA-SUBA / GERB-SUBB / GERC-SUBC
            return coluna

    raise ValueError(
        f"Não foi encontrado sinal correspondente à fase {fase}."
    )
# Importa a função original para realizar uma primeira filtragem das variáveis do PL4
from f_selecionavariaveisATP import seleciona_variaveis

# Seleciona somente os sinais correspondentes aos medidores e à subestação
def selecionar_sinais_medidores(pl4, incluir_time=True):
    """
    Seleciona tensões e correntes dos medidores Mxx e da SUB.

    Retorna:
        dict:
            time, quando solicitado;
            tensões V-node dos medidores Mxx;
            tensões V-node da SUB;
            correntes Nxx -> Mxx;
            correntes GER -> SUB.
    """

    # ============================================================
    # Seleção inicial utlizando a função original
    # ============================================================
    variaveis_filtradas = seleciona_variaveis(
        pl4,
        filtros=["M_NUM", "N", "SUB", "GER"],
        incluir_time=incluir_time
    )

    # ============================================================
    # Refinamento para manter somente medidores + SUB
    # ============================================================
    # Dicionário que armazenará apenas os sinais necessários
    variaveis_selecionadas = {}

    for nome, dados in variaveis_filtradas.items():
        # Mantém o vetor de tempo
        if nome == "time":
            variaveis_selecionadas[nome] = dados
            continue

        # Mantém as tensões nodais dos medidores identificados por Mxx
        if nome.startswith("M") and "(V-node)" in nome:
            variaveis_selecionadas[nome] = dados
            continue

        # Mantém as tensões das três fases da subestação
        if nome in {
            "SUBA- (V-node)",
            "SUBB- (V-node)",
            "SUBC- (V-node)"
        }:
            variaveis_selecionadas[nome] = dados
            continue

        # Mantém as correntes dos ramos associados aos medidores Nxx -> Mxx
        if "(I-bran)" in nome and nome.startswith("N"):
            ramo = nome.split(" (")[0]

            if "-" in ramo:
                origem, destino = ramo.split("-", 1)

                if (
                    origem.startswith("N")
                    and destino.startswith("M")
                    and origem[1:] == destino[1:]
                ):
                    variaveis_selecionadas[nome] = dados
                    continue

        # Mantém as correntes das três fases entre o gerador e a subestação
        if nome in {
            "GERA-SUBA (I-bran)",
            "GERB-SUBB (I-bran)",
            "GERC-SUBC (I-bran)"
        }:
            variaveis_selecionadas[nome] = dados

    # ============================================================
    # Padronização dos nomes das tensões
    # ============================================================
    # Dicionário que armazenará os nomes padronizados
    variaveis_renomeadas = {}

    # Remove o hífen residual presente nos nomes das tensões nodais
    for nome, dados in variaveis_selecionadas.items():
        if nome.endswith("- (V-node)"):
            # Por exemplo: "SUBA- (V-node)" -> "SUBA (V-node)"
            nome = nome.replace("- (V-node)", " (V-node)")

        # Armazena a variável utilizando o nome padronizado
        variaveis_renomeadas[nome] = dados

    # Retorna somente os sinais selecionados, com os nomes padronizados
    return variaveis_renomeadas
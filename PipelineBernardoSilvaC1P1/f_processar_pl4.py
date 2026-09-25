from pathlib import Path
from f_selecionar_sinais_medidores import selecionar_sinais_medidores
from scipy.signal import resample_poly
import re
import numpy as np

FREQUENCIA_REDE = 60.0 # frequência fundamental da rede elétrica, em Hz
CICLOS_ANTES = 20 # quantidade de ciclos que serão mantidos antes do início da FAI
CICLOS_DEPOIS = 20 # quantidade de ciclos que serão mantidos depois do início da FAI
AMOSTRAS_POR_CICLO_DESTINO = 64 # quantidade desejada de amostras por ciclo após a subamostragem

# ============================================================
# Extração do instante de início da FAI - TTOQ
# ============================================================
# Obtém o valor de TTOQ diretamente das definições presentes no cartão ATP
def extrair_ttoq(arquivo_atp):
    arquivo_atp = Path(arquivo_atp)

    # Lê o conteúdo textual do cartão ATP
    conteudo = arquivo_atp.read_text(
        encoding="latin-1",
        errors="ignore"
    )

    # Localiza no cartão a definição do parâmetro TRUP
    match_trup = re.search(
        r"\bTRUP\s*=\s*([0-9.+\-Ee]+)",
        conteudo,
        flags=re.IGNORECASE
    )

    # Localiza a definição de TTOQ no formato TTOQ = TRUP + valor
    match_ttoq = re.search(
        r"\bTTOQ\s*=\s*TRUP\s*\+\s*([0-9.+\-Ee]+)",
        conteudo,
        flags=re.IGNORECASE
    )

    if not match_trup:
        raise ValueError(
            f"Não foi possível localizar TRUP em {arquivo_atp.name}."
        )

    if not match_ttoq:
        raise ValueError(
            f"Não foi possível localizar TTOQ em {arquivo_atp.name}."
        )

    # Converte o valor de TRUP encontrado no cartão para número real
    trup = float(match_trup.group(1))
    # Obtém o incremento definido na expressão TTOQ = TRUP + incremento
    incremento_ttoq = float(match_ttoq.group(1))

    # Calcula e retorna o instante absoluto de TTOQ
    return trup + incremento_ttoq

# ============================================================
# Processamento do arquivo PL4
# ============================================================
# Seleciona, recorta e reamostra os sinais obtidos na simulação
def processar_pl4(pl4, arquivo_atp):
    # ============================================================
    # Seleção dos sinais
    # ============================================================
    # Mantém somente o tempo, as tensões e correntes dos medidores e da SUB
    variaveis = selecionar_sinais_medidores(
        pl4,
        incluir_time=True
    )

    # Converte o vetor de tempo para um array NumPy
    tempo = np.asarray(variaveis["time"])

    # Obtém os nomes de todos os sinais elétricos, excluindo o tempo
    nomes_sinais = [
        nome
        for nome in variaveis
        if nome != "time"
    ]

    # Agrupa os sinais em uma matriz, com um sinal em cada linha e as amostras temporais distribuídas nas colunas
    dados = np.vstack([
        np.asarray(variaveis[nome])
        for nome in nomes_sinais
    ])

    # ============================================================
    # Informações temporais
    # ============================================================
    if len(tempo) < 2:
        raise ValueError(
            "O arquivo PL4 não possui amostras suficientes."
        )

    # Calcula o tempo de amostragem médio
    delta_t = float(np.mean(np.diff(tempo)))
    # Calcula a frequência de amostragem original do PL4
    frequencia_amostragem = 1.0 / delta_t
    # Calcula as amostras por ciclo
    amostras_por_ciclo = (
        frequencia_amostragem / FREQUENCIA_REDE
    )
    # Arredonda o valor das amostras por ciclo
    amostras_por_ciclo_inteiro = round(
        amostras_por_ciclo
    )

    if not np.isclose(
        amostras_por_ciclo,
        amostras_por_ciclo_inteiro,
        atol=1e-3
    ):
        raise ValueError(
            "A quantidade de amostras por ciclo não está "
            "suficientemente próxima de um valor inteiro."
        )

    # ============================================================
    # TTOQ
    # ============================================================
    # Obtém do cartão ATP o instante definido para o início da FAI
    ttoq = extrair_ttoq(arquivo_atp)

    # Localiza no vetor de tempo a amostra mais próxima de TTOQ
    indice_evento = int(
        np.argmin(np.abs(tempo - ttoq))
    )

    # ============================================================
    # Recorte
    # ============================================================
    # Calcula a quantidade de amostras correspondente aos 20 ciclos anteriores ao TTOQ
    n_antes = (
        CICLOS_ANTES
        * amostras_por_ciclo_inteiro
    )
    # Calcula a quantidade de amostras correspondente aos # 20 ciclos posteriores ao TTOQ
    n_depois = (
        CICLOS_DEPOIS
        * amostras_por_ciclo_inteiro
    )

    # Define o índice inicial do recorte
    indice_inicio = indice_evento - n_antes
    # Define o índice final do recorte
    indice_fim = indice_evento + n_depois

    if indice_inicio < 0:
        raise ValueError(
            "Não existem amostras suficientes antes de TTOQ."
        )

    if indice_fim > len(tempo):
        raise ValueError(
            "Não existem amostras suficientes depois de TTOQ."
        )

    # Recorta todos os sinais utilizando a mesma janela temporal
    dados_recortados = dados[
        :,
        indice_inicio:indice_fim
    ]

    # Calcula a quantidade total de amostras esperada após o recorte
    total_esperado = n_antes + n_depois

    # Verifica se o recorte contém exatamente a quantidade esperada de amostras para 40 ciclos
    if dados_recortados.shape[1] != total_esperado:
        raise ValueError(
            "Quantidade incorreta de amostras após o recorte."
        )

    # ============================================================
    # Reamostragem
    # ============================================================
    if (
        amostras_por_ciclo_inteiro
        % AMOSTRAS_POR_CICLO_DESTINO
        != 0
    ):
        raise ValueError(
            "A quantidade original de amostras por ciclo "
            "não é divisível pela quantidade desejada."
        )

    # Calcula o fator de redução da frequência de amostragem
    fator_reducao = (
        amostras_por_ciclo_inteiro
        // AMOSTRAS_POR_CICLO_DESTINO
    )

    # Reamostra todos os sinais
    dados_reamostrados = resample_poly( # essa função também aplica anti-aliasing
        dados_recortados,
        up=1,
        down=fator_reducao, # 4
        axis=1
    )

    # Calcula a quantidade final esperada de amostras
    total_final = (
        (CICLOS_ANTES + CICLOS_DEPOIS)
        * AMOSTRAS_POR_CICLO_DESTINO
    )

    if dados_reamostrados.shape[1] != total_final:
        raise ValueError(
            "Quantidade incorreta de amostras após "
            "a reamostragem."
        )

    # ============================================================
    # Tempo relativo a TTOQ
    # ============================================================
    # Calcula a frequência de amostragem final
    frequencia_final = (
        FREQUENCIA_REDE
        * AMOSTRAS_POR_CICLO_DESTINO
    )

    # Calcula a posição de TTOQ no vetor final
    indice_evento_final = (
        CICLOS_ANTES
        * AMOSTRAS_POR_CICLO_DESTINO
    )

    # Cria um novo vetor de tempo relativo a TTOQ
    tempo_relativo = (
        np.arange(total_final) - indice_evento_final
    ) / frequencia_final

    # Verifica se TTOQ está corretamente posicionado em t = 0
    if tempo_relativo[indice_evento_final] != 0:
        raise ValueError(
            "TTOQ não corresponde a t = 0 no vetor final."
        )

    # Retorna os sinais processados e as principais informações temporais utilizadas no processamento
    return {
        "time": tempo_relativo,
        "dados": dados_reamostrados,
        "nomes_sinais": nomes_sinais,
        "ttoq": ttoq,
        "frequencia_amostragem_original": frequencia_amostragem,
        "frequencia_amostragem_final": frequencia_final,
        "indice_evento": indice_evento_final
    }
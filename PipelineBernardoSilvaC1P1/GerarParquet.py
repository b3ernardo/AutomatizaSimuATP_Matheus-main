from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import time
import uuid
import sys
import pandas as pd
import numpy as np
import psutil

# ============================================================
# Diretórios para importação
# ============================================================
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_REPOSITORIO_RAIZ = DIRETORIO_PIPELINE.parent

# Necessário para importar funções localizadas fora da pasta da pipeline
sys.path.insert(0, str(DIRETORIO_REPOSITORIO_RAIZ))

from f_readpl4 import readpl4
from f_processa_pl4 import processar_pl4

# ============================================================
# Configurações
# ============================================================
NUM_THREADS = 4 # define o número de threads
LIMITE_TESTE = None # define até quantos cartões vão rodar

# ============================================================
# Diretórios
# ============================================================
DIRETORIO_CARTOES = (
    DIRETORIO_PIPELINE
    / "TodosCartoesRodadosSemRetirarCargas"
    / "CartoesATP"
)
RUNATP_PATH = DIRETORIO_REPOSITORIO_RAIZ / "runATP.exe" # executável que inicia as simulações do ATP
DIRETORIO_TEMP = DIRETORIO_PIPELINE / "temp_atp"
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets"
DIRETORIO_FALHAS = DIRETORIO_PIPELINE / "CartoesNaoRodados"

# ============================================================
# Validação dos diretórios
# ============================================================
if not RUNATP_PATH.exists():
    raise FileNotFoundError(
        f"Executável runATP.exe não encontrado em: {RUNATP_PATH}"
    )

# Cria os diretórios de trabalho caso ainda não existam
DIRETORIO_TEMP.mkdir(exist_ok=True)
DIRETORIO_PARQUETS.mkdir(exist_ok=True)
DIRETORIO_FALHAS.mkdir(exist_ok=True)

# ============================================================
# Seleção dos cartões P1/C1 pendentes
# ============================================================
# Busca e ordena todos os cartões ATP correspondentes ao P1/C1
todos_arquivos_p1_c1 = sorted(
    arquivo
    for arquivo in DIRETORIO_CARTOES.glob("*.atp")
    if "_p1_carregamento1_" in arquivo.stem.lower()
)

if not todos_arquivos_p1_c1:
    raise FileNotFoundError("Nenhum cartão P1/C1 foi encontrado.")

# Seleciona apenas os cartões que ainda não possuem um Parquet correspondente
arquivos_p1_c1 = [
    arquivo
    for arquivo in todos_arquivos_p1_c1
    if not (
        DIRETORIO_PARQUETS
        / f"{arquivo.stem}.parquet"
    ).exists()
]

# Calcula a quantidade total de cartões e quantos já foram processados
total_encontrado = len(todos_arquivos_p1_c1)
total_existente = total_encontrado - len(arquivos_p1_c1)

# Limita a quantidade de cartões processados quando estiver em modo de teste
if LIMITE_TESTE is not None:
    arquivos_p1_c1 = arquivos_p1_c1[:LIMITE_TESTE]

print(f"Total de cartões P1/C1 encontrados: {total_encontrado}")
print(f"Parquets já existentes: {total_existente}")
print(f"Cartões pendentes: {len(arquivos_p1_c1)}")
print(f"Execuções simultâneas: {NUM_THREADS}")

# ============================================================
# Conta quantas instâncias do simulador ATP estão em execução
# ============================================================
def contar_tpbig():
    quantidade = 0

    # Percorre os processos ativos no sistema operacional
    for processo in psutil.process_iter(["name"]):
        try:
            nome = processo.info["name"]

            # Incrementa o contador quando encontra uma instância do ATP
            if nome and nome.lower() == "tpbigm.exe":
                quantidade += 1

        # Ignora processos encerrados ou sem permissão de acesso
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Retorna a quantidade de instâncias do ATP em execução
    return quantidade

# ============================================================
# Processa o cartão ATP e gera o Parquet
# ============================================================
def processar_cartao(cartao: Path):
    # Gera um identificador único para isolar a execução do cartão
    identificador = uuid.uuid4().hex[:8]

    # Define um diretório temporário exclusivo para a execução
    diretorio_execucao = (
        DIRETORIO_TEMP
        / f"{cartao.stem}_{identificador}"
    )

    try:
        # ====================================================
        # Preparação do diretório temporário
        # ====================================================
        # Cria o diretório temporário da execução
        diretorio_execucao.mkdir(
            parents=True,
            exist_ok=False,
        )

        # Define o caminho da cópia temporária do cartão ATP
        cartao_copia = diretorio_execucao / cartao.name

        # Copia o cartão ATP original para o diretório temporário
        shutil.copy2(
            cartao,
            cartao_copia,
        )

        # ====================================================
        # Execução do ATP
        # ====================================================
        print(f"Iniciando ATP: {cartao.name}")

        cmd = [str(RUNATP_PATH), cartao_copia.name]

        # Inicia o runATP.exe no diretório temporário do cartão
        processo = subprocess.Popen(
            cmd,
            cwd=str(diretorio_execucao),
            shell=False,
        )

        # Aguarda o processo do ATP iniciar
        tpbig_iniciado = False

        while processo.poll() is None:
            try:
                # Obtém o processo runATP.exe iniciado pelo Python
                processo_runatp = psutil.Process(processo.pid)

                # Obtém os processos iniciados pelo runATP.exe
                filhos = processo_runatp.children(recursive=True)

                # Verifica se o simulador tpbig.exe foi iniciado
                if any(
                    filho.name().lower() == "tpbig.exe"
                    for filho in filhos
                ):
                    tpbig_iniciado = True
                    break

            # Ignora processos encerrados ou sem permissão de acesso
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                pass

            # Aguarda antes de realizar uma nova verificação
            time.sleep(0.5)

        # Interrompe o processamento caso o simulador não tenha sido iniciado
        if not tpbig_iniciado:
            raise RuntimeError(
                "O processo tpbig.exe não foi identificado."
            )

        # Aguarda o tpbig.exe terminar a simulação
        while processo.poll() is None:
            try:
                processo_runatp = psutil.Process(processo.pid)

                filhos = processo_runatp.children(recursive=True)

                tpbig_ativo = any(
                    filho.name().lower() == "tpbig.exe"
                    for filho in filhos
                )

                if not tpbig_ativo:
                    break

            except psutil.NoSuchProcess:
                break

            except psutil.AccessDenied:
                pass

            time.sleep(0.5)

        # O tpbig.exe terminou a simulação e o runATP.exe permanece apenas aguardando uma tecla
        if processo.poll() is None:
            print(
                f"Simulação concluída. Encerrando runATP.exe: {cartao.name}"
            )

            # Encerra somente a instância do runATP.exe associada a este cartão
            processo.terminate()
            processo.wait()

        print(f"ATP finalizado: {cartao.name}")

        # ====================================================
        # Localização do PL4 temporário
        # ====================================================
        pl4_temporario = cartao_copia.with_suffix(".pl4")

        # Verifica se o ATP gerou o arquivo PL4 no diretório temporário
        if not pl4_temporario.exists():
            raise FileNotFoundError(
                "A execução terminou, mas nenhum arquivo PL4 foi encontrado."
            )

        print(f"PL4 gerado: {pl4_temporario.name}")

        # ====================================================
        # Lê os sinais armazenados no arquivo PL4
        # ====================================================
        resultado_pl4 = readpl4(
            str(pl4_temporario)
        )

        print(f"PL4 lido: {pl4_temporario.name}")

        # ====================================================
        # Processamento dos sinais
        # ====================================================
        # Seleciona os medidores, recorta e reamostra os sinais
        resultado_processado = processar_pl4(
            resultado_pl4,
            cartao_copia,
        )

        print(f"Sinais processados: {cartao.name}")

        # ====================================================
        # Criação do DataFrame
        # ====================================================
        # Organiza os sinais em linhas de tempo e colunas de variáveis
        df = pd.DataFrame(
            resultado_processado["dados"].T,
            columns=resultado_processado["nomes_sinais"],
        )

        # Insere o vetor de tempo como primeira coluna
        df.insert(
            0,
            "time",
            resultado_processado["time"],
        )

        # ====================================================
        # Salvamento do Parquet
        # ====================================================
        # Define o caminho do Parquet correspondente ao cartão ATP
        caminho_parquet = (
            DIRETORIO_PARQUETS
            / f"{cartao.stem}.parquet"
        )

        # Salva os sinais processados no formato Parquet
        df.to_parquet(
            caminho_parquet,
            index=False,
        )

        # ====================================================
        # Validação do Parquet
        # ====================================================
        # Verifica se o arquivo Parquet foi efetivamente criado
        if not caminho_parquet.exists():
            raise FileNotFoundError(
                "O processamento terminou, mas o Parquet não foi criado."
            )

        # Verifica as 2560 amostras esperadas: 40 ciclos × 64 amostras/ciclo
        if len(df) != 2560:
            raise ValueError(
                f"Número inesperado de linhas no Parquet: {len(df)}."
            )

        # Verifica as 165 colunas esperadas: tempo + 164 sinais elétricos
        if len(df.columns) != 165:
            raise ValueError(
                f"Número inesperado de colunas no Parquet: {len(df.columns)}."
            )

        # Define o índice correspondente ao TTOQ após os 20 ciclos anteriores
        indice_evento = 20 * 64

        # Verifica se o instante do TTOQ está corretamente posicionado em t = 0
        if not np.isclose(
            df.loc[indice_evento, "time"],
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "TTOQ não está posicionado em t = 0 no Parquet."
            )

        print(f"Parquet salvo: {caminho_parquet.name}")

        # Retorna o resultado da execução bem-sucedida
        return {
            "arquivo": cartao.name,
            "sucesso": True,
            "erro": None,
        }

    except Exception as erro:
        # ====================================================
        # Registro do cartão que apresentou falha
        # ====================================================
        # Define o destino do cartão que apresentou erro
        destino_falha = DIRETORIO_FALHAS / cartao.name

        # Copia o cartão com falha para o diretório de análise
        if not destino_falha.exists():
            shutil.copy2(
                cartao,
                destino_falha,
            )

        # Retorna as informações da execução que apresentou erro
        return {
            "arquivo": cartao.name,
            "sucesso": False,
            "erro": str(erro),
        }

    finally:
        # ====================================================
        # Limpeza dos arquivos temporários
        # ====================================================
        if diretorio_execucao.exists():
            shutil.rmtree(
                diretorio_execucao,
                ignore_errors=True,
            )

# ============================================================
# Execução em lote
# ============================================================
# Inicializa os contadores de cartões processados com sucesso e com falha
sucessos = 0
falhas = 0

# Cria o conjunto de threads para execução simultânea dos cartões
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futuros = []

    # Percorre os cartões ATP pendentes
    for cartao in arquivos_p1_c1:
        # Garante que não ultrapasse o número máximo de processamentos ao mesmo tempo
        while contar_tpbig() >= NUM_THREADS:
            time.sleep(0.5)

        # Se tem thread disponível, envia o cartão para processamento em uma das threads
        futuro = executor.submit(
            processar_cartao,
            cartao,
        )

        futuros.append(futuro)

    # Percorre as tarefas conforme elas são finalizadas
    for indice, futuro in enumerate(
        as_completed(futuros),
        start=1,
    ):
        # Obtém o resultado retornado por processar_cartao()
        resultado = futuro.result()

        # Contabiliza e exibe as execuções bem-sucedidas
        if resultado["sucesso"]:
            sucessos += 1

            print(
                f"[{indice}/{len(arquivos_p1_c1)}] "
                f"{resultado['arquivo']} -> OK"
            )
        # Contabiliza e exibe as execuções que apresentaram erro
        else:
            falhas += 1

            print(
                f"[{indice}/{len(arquivos_p1_c1)}] "
                f"{resultado['arquivo']} -> ERRO"
            )

            print(
                f"    {resultado['erro']}"
            )

# ============================================================
# Resumo após todos os processamentos
# ============================================================
print("\nProcessamento finalizado.")

print(f"Sucessos: {sucessos}")
print(f"Falhas: {falhas}")

print(
    f"Parquets existentes: "
    f"{len(list(DIRETORIO_PARQUETS.glob('*.parquet')))}"
)
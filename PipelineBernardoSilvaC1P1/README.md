# Pipeline de processamento e classificação de faltas de alta impedância

Pipeline desenvolvido para a localização de faltas de alta impedância (FAIs) em um sistema IEEE de 34 barras modificado com geração distribuída, utilizando exclusivamente medições de tensão e corrente da subestação (`SUB`).

O fluxo contempla a execução das simulações no ATP, o pré-processamento dos sinais, a extração e seleção de atributos, a regionalização do alimentador e o treinamento, a otimização e a avaliação dos classificadores.

O estudo utiliza 1.370 simulações P1/C1 (condições nominais de potência da geração distribuída e de carregamento), sendo 810 com GD e 560 sem GD. As faltas ocorrem nas barras 2 a 34. São consideradas as janelas `(0, 5)`, `(0, 10)` e `(0, 20)`, correspondentes aos primeiros 5, 10 e 20 ciclos após o contato do condutor com o solo (TTOQ).

## 1. Ordem de execução

O trabalho foi desenvolvido em duas etapas: análise exploratória, com 64 amostras por ciclo e harmônicas até a 20ª ordem, e configuração final, com 16 amostras por ciclo e somente a componente fundamental. **Os mesmos scripts `GerarParquet.py` e `ProcessarAtributos.py` são utilizados nas duas etapas.** A taxa de amostragem, a ordem harmônica e os diretórios são definidos nas configurações do código; não há scripts separados com sufixo `16`.

| Ordem | Script | Entrada | Saída / finalidade |
|---|---|---|---|
| 1 | `AnalisarArquivosCartoesATP.py` (opcional) | Cartões ATP | Inspeção da base de simulações |
| 2 | `GerarParquet.py` (configuração inicial) | Cartões ATP | `Parquets/` — sinais a 64 amostras/ciclo |
| 3 | `ProcessarAtributos.py` (configuração inicial) | `Parquets/` | `DatasetAtributos/` — 2.160 atributos candidatos |
| 4 | `SelecaoAtributos.py` | `DatasetAtributos/` | Rankings ANOVA e RF e comparação dos atributos |
| 5 | `GerarParquet.py` (reconfigurado) | Cartões ATP | `Parquets16/` — sinais a 16 amostras/ciclo |
| 6 | `ProcessarAtributos.py` (reconfigurado) | `Parquets16/` | `DatasetAtributos16/` — 108 atributos candidatos |
| 7 | `Regionalizacoes.py` (módulo) | Definições das regiões | Dicionário `REGIONALIZACOES`, importado na classificação |
| 8 | `Classificadores.py` | `DatasetAtributos16/` e `Regionalizacoes.py` | Hiperparâmetros JSON, métricas no terminal e gráficos PDF |

**Importante:** `Regionalizacoes.py` e os módulos `f_*.py` não precisam ser executados separadamente. O script de análise de cartões é opcional e não é uma dependência dos classificadores.

### Execução rápida — configuração final

Se `DatasetAtributos16/` já contém os três datasets de atributos, basta executar:

```powershell
python .\Classificadores.py
```

Se existem apenas os sinais processados em `Parquets16/`, configure `ProcessarAtributos.py` conforme a seção 5.2 e execute:

```powershell
python .\ProcessarAtributos.py
python .\Classificadores.py
```

Se os datasets finais ainda não existem e os Parquets de 16 amostras/ciclo precisam ser gerados, configure `GerarParquet.py`, `f_processar_pl4.py` e `ProcessarAtributos.py` conforme as seções 5.1 e 5.2. Em seguida, execute:

```powershell
python .\GerarParquet.py
python .\ProcessarAtributos.py
python .\Classificadores.py
```

A etapa exploratória é necessária para reproduzir a seleção de atributos apresentada no trabalho, mas não é necessário refazê-la para treinar os classificadores com os seis atributos já definidos.

> O sufixo `16` identifica os diretórios de dados da configuração final, não os scripts Python. Mantenha os dados das duas configurações em pastas distintas e confira os parâmetros antes de executar.

## 2. Pré-requisitos e configuração do ambiente

- Windows e Visual Studio Code com a extensão Python da Microsoft.
- Python 3 em versão compatível com as dependências do projeto. A versão exata do ambiente original deve ser conferida com `python --version`.
- Repositório completo, incluindo os módulos auxiliares e os dados de entrada da etapa desejada.
- ATP instalado/configurado e `runATP.exe` disponível na pasta superior à pipeline, somente se for necessário gerar novos Parquets.

Abra a pasta `PipelineBernardoSilvaC1P1` no VS Code e selecione **Terminal > Novo Terminal**. Os comandos deste README devem ser executados nessa pasta.

### Ambiente virtual

Verifique a versão do Python e crie a `.venv`, caso ainda não exista:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o terminal já apresenta `(.venv)`, não é necessário ativar o ambiente novamente. No VS Code, utilize `Ctrl+Shift+P` > `Python: Select Interpreter` para selecionar o interpretador da `.venv`.

Caso o PowerShell bloqueie a ativação, a seguinte configuração vale somente para a sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Dependências

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow scipy scikit-learn matplotlib psutil optuna xgboost
```

| Biblioteca | Finalidade |
|---|---|
| `numpy` | Operações numéricas e tratamento de sinais |
| `pandas`, `pyarrow` | Organização dos dados e leitura/gravação de Parquets |
| `scipy` | Processamento de sinais, incluindo reamostragem |
| `scikit-learn` | Seleção de atributos, classificadores, validação cruzada, escalonamento e métricas |
| `optuna` | Otimização de hiperparâmetros |
| `xgboost` | Classificador XGBoost |
| `matplotlib` | Geração dos gráficos |
| `psutil` | Inspeção dos processos ATP |

Se houver um `requirements.txt` atualizado no repositório, utilize-o para reproduzir as versões do ambiente:

```powershell
python -m pip install -r requirements.txt
```

As versões de `scikit-learn`, Optuna e XGBoost podem afetar a compatibilidade dos hiperparâmetros salvos. Para registrar as versões utilizadas em uma execução, pode-se usar `python -m pip freeze`.

## 3. Estrutura do projeto

Estrutura lógica esperada, com os scripts principais organizados na ordem do fluxo:

```text
AutomatizaSimuATP_Matheus-main/
├── runATP.exe
├── f_readpl4.py
├── f_processar_pl4.py
└── PipelineBernardoSilvaC1P1/
    ├── README.md
    ├── AnalisarArquivosCartoesATP.py    # Opcional
    ├── GerarParquet.py                 # Geração de sinais: 64 ou 16 amostras/ciclo
    ├── ProcessarAtributos.py           # Extração: H1–H20 ou H1
    ├── SelecaoAtributos.py
    ├── Regionalizacoes.py
    ├── Classificadores.py
    ├── f_recortar_janela.py
    ├── f_selecionar_sinais_ponto.py
    ├── f_selecionar_sinais_medidores.py
    ├── f_preparar_sinais.py
    ├── f_extrair_harmonicas.py
    ├── f_calcular_parametros.py
    ├── f_calcular_estatisticas.py
    ├── TodosCartoesRodadosSemRetirarCargas/  # Cartões .atp de entrada
    ├── Parquets/                       # Sinais da etapa exploratória
    ├── DatasetAtributos/               # Atributos da etapa exploratória
    ├── Parquets16/                     # Sinais da configuração final
    ├── DatasetAtributos16/             # Atributos da configuração final
    ├── Hiperparametros/                # Caches JSON por classificador
    ├── Graficos/                       # Figuras PDF da avaliação
    ├── temp_atp/                       # Arquivos temporários das simulações
    ├── CartoesNaoRodados/               # Tratamento de falhas na geração
    └── ParquetsDiagnostico/             # Opcional, se existente
```

**Atenção:** os caminhos reais devem coincidir com as constantes dos scripts. Na configuração de geração compartilhada, `DIRETORIO_CARTOES` aponta diretamente para `TodosCartoesRodadosSemRetirarCargas/`, com busca não recursiva por `*.atp`. Se os cartões estiverem na subpasta `CartoesATP/`, ajuste esse caminho. Os módulos `f_readpl4.py` e `f_processar_pl4.py` são importados a partir da raiz do repositório.

## 4. Etapa exploratória — seleção de atributos

### 4.1. Gerar os sinais a 64 amostras por ciclo — `GerarParquet.py`

Configure o script para a etapa inicial:

```python
AMOSTRAS_POR_CICLO = 64
NUM_THREADS = 4
LIMITE_TESTE = None
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets"
```

Confira também a configuração de reamostragem em `f_processar_pl4.py`, que deve produzir 64 amostras por ciclo (redução de 256 para 64, fator 4). No código compartilhado, `processar_pl4` é chamado sem receber `AMOSTRAS_POR_CICLO` como argumento, portanto a constante do gerador, utilizada na validação, não garante sozinha a taxa dos sinais processados.

```powershell
python .\GerarParquet.py
```

O script seleciona os cartões ATP cujos nomes contêm `_p1_carregamento1_`, ignora os que já possuem Parquet correspondente e executa as simulações pendentes em paralelo (`NUM_THREADS = 4`). Os arquivos PL4 são lidos e processados, preservando os nomes originais dos cartões nos Parquets.

O pré-processamento seleciona os 164 sinais elétricos (82 de tensão e 82 de corrente), recorta 20 ciclos antes e 20 ciclos após o TTOQ, realiza a reamostragem polifásica com filtro antialiasing e alinha o instante do contato com o solo em `t = 0`.

Na configuração inicial, os sinais são reduzidos de 256 para 64 amostras por ciclo de 60 Hz. Cada Parquet possui 2.560 linhas e 165 colunas (tempo + 164 sinais). O TTOQ está no índice 1.280, considerando indexação iniciada em zero.

Saída: `Parquets/`, um arquivo por simulação.

### 4.2. Extrair os atributos iniciais — `ProcessarAtributos.py`

```powershell
python .\ProcessarAtributos.py
```

Configuração necessária para reproduzir a análise inicial:

```python
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets"
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos"
JANELAS = [(0, 5), (0, 10), (0, 20)]
AMOSTRAS_POR_CICLO = 64
ORDEM_HARMONICA_MAXIMA = 20
PONTO_MEDICAO = "SUB"
```

Para cada simulação e janela, o script identifica a barra e a fase faltosa pelo nome do arquivo, recorta os sinais, seleciona a subestação e extrai atributos de duas representações: soma das três fases e fase faltosa. A FFT é aplicada ciclo a ciclo.

São calculados nove parâmetros para cada harmônica: energia espectral, módulo e ângulo da tensão; energia espectral, módulo e ângulo da corrente; diferença angular, módulo da impedância e métrica de potência ativa. Cada parâmetro é resumido por seis estatísticas: média, máximo, mínimo, desvio padrão, variância e amplitude.

Total por simulação e janela: `2 × 20 × 9 × 6 = 2.160` atributos, além de `arquivo`, `fase_faltosa` e `barra`.

```text
DatasetAtributos/
├── atributos_0_5.parquet
├── atributos_0_10.parquet
└── atributos_0_20.parquet
```

Cada dataset completo esperado contém 1.370 linhas e 2.163 colunas. A regionalização não é realizada nessa etapa: a barra original é preservada como metadado.

### 4.3. Avaliar a relevância dos atributos — `SelecaoAtributos.py`

Para reproduzir a seleção apresentada no artigo, configure:

```python
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos"
JANELAS = [(0, 5), (0, 10), (0, 20)]
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_N = 40
```

```powershell
python .\SelecaoAtributos.py
```

O script utiliza a barra de ocorrência da falta como variável resposta (33 classes). A divisão treino/teste é estratificada pela barra, com 1.096 simulações para treinamento e 274 para teste. A mesma divisão é aplicada a todas as janelas. Os rankings são calculados exclusivamente com as amostras de treinamento.

- **ANOVA:** método de filtro que avalia a capacidade discriminativa individual dos atributos por meio da estatística F; atributos constantes são tratados antes do ranking.
- **Random Forest:** método embutido que utiliza a importância dos atributos baseada na redução de impureza das árvores de decisão.

Os hiperparâmetros de RF são definidos no próprio script por janela, independentemente dos hiperparâmetros da etapa de classificação regional:

| Janela | Árvores | Profundidade máxima | Mínimo por divisão | Mínimo por folha | Atributos por divisão |
|---|---:|---|---:|---:|---|
| `(0, 5)` | 100 | 40 | 2 | 1 | Todos (`None`) |
| `(0, 10)` | 100 | Ilimitada (`None`) | 3 | 1 | Todos (`None`) |
| `(0, 20)` | 200 | 40 | 5 | 1 | Todos (`None`) |

O código apresenta os rankings, a comparação dos 40 primeiros atributos de cada método e as posições dos seis atributos de interesse. Os resultados são exibidos no terminal; a versão compartilhada não grava um arquivo de seleção para os classificadores.

A análise inicial evidenciou a predominância da componente fundamental (H1). Para dispensar o conhecimento prévio da fase faltosa durante a classificação, foram escolhidos atributos da soma das três fases. A seleção final foi definida com base nos rankings e nesse critério físico, e não pela escolha automática dos seis primeiros colocados.

> A versão mais recente de `SelecaoAtributos.py` compartilhada aponta para `DatasetAtributos16/`. Para reproduzir os rankings originais de 2.160 atributos, altere `DIRETORIO_DATASETS` para `DatasetAtributos/`. A análise sobre o dataset final de 108 atributos é opcional e não substitui a análise exploratória original.

## 5. Configuração final dos dados

A restrição à componente fundamental de 60 Hz permite reduzir a taxa de amostragem. As configurações inicial e final são:

| Característica | Inicial | Final |
|---|---:|---:|
| Amostras por ciclo | 64 | 16 |
| Frequência de amostragem | 3.840 Hz | 960 Hz |
| Frequência de Nyquist | 1.920 Hz | 480 Hz |
| Componentes harmônicas | H1–H20 | H1 |
| Amostras por sinal (40 ciclos) | 2.560 | 640 |
| Colunas dos Parquets de sinais | 165 | 165 |
| Atributos extraídos por janela | 2.160 | 108 |
| Atributos utilizados na classificação | — | 6 |

### 5.1. Gerar os sinais finais — `GerarParquet.py`

Reutilize o mesmo script da etapa 4.1, alterando as configurações:

```python
AMOSTRAS_POR_CICLO = 16
NUM_THREADS = 4
LIMITE_TESTE = None
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets16"
```

Ajuste a reamostragem em `f_processar_pl4.py` para produzir 16 amostras por ciclo a partir dos sinais originais (redução de 256 para 16, fator 16). Em seguida, execute:

```powershell
python .\GerarParquet.py
```

O script utiliza o mesmo conjunto de cartões P1/C1, com `AMOSTRAS_POR_CICLO = 16`, `NUM_THREADS = 4` e `LIMITE_TESTE = None`. A saída é `Parquets16/`.

Cada Parquet deve conter 640 linhas e 165 colunas, com o TTOQ no índice 320 (tempo igual a zero). O processamento deve reamostrar corretamente os sinais originais para 16 amostras por ciclo; alterar somente a constante no script gerador não altera automaticamente a taxa produzida por `f_processar_pl4.py`.

**Atenção:** as duas configurações utilizam o mesmo módulo `f_processar_pl4.py`. Antes de executar `GerarParquet.py`, confira se a função auxiliar está configurada para a taxa correspondente. Preserve a filtragem antialiasing e mantenha as duas saídas em pastas distintas. Na configuração de 16 amostras/ciclo, o gerador reexecuta o ATP quando o Parquet correspondente está ausente; ele não lê os Parquets de 64 amostras/ciclo como entrada.

### 5.2. Extrair os atributos finais — `ProcessarAtributos.py`

Reutilize o mesmo script da etapa 4.2, alterando as configurações:

```python
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets16"
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos16"
JANELAS = [(0, 5), (0, 10), (0, 20)]
AMOSTRAS_POR_CICLO = 16
ORDEM_HARMONICA_MAXIMA = 1
PONTO_MEDICAO = "SUB"
```

Após configurar, execute:

```powershell
python .\ProcessarAtributos.py
```

O procedimento de extração é o mesmo da etapa inicial, porém limitado à H1. O código ainda extrai as duas representações e todas as estatísticas: `2 × 1 × 9 × 6 = 108` atributos candidatos. Cada dataset completo esperado contém 1.370 linhas e 111 colunas (108 atributos + 3 metadados).

```text
DatasetAtributos16/
├── atributos_0_5.parquet
├── atributos_0_10.parquet
└── atributos_0_20.parquet
```

Os seis atributos efetivamente utilizados na classificação são selecionados por nome em `Classificadores.py`:

```python
ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]
```

São os valores máximo, médio e mínimo dos módulos da componente fundamental da soma das correntes e da soma das tensões das três fases.

## 6. Regionalização do alimentador — `Regionalizacoes.py`

O módulo contém os dicionários `REGIOES_1` a `REGIOES_33`, reunidos em `REGIONALIZACOES`. Os agrupamentos foram definidos empiricamente com base na topologia do alimentador.

`Classificadores.py` importa esse dicionário e cria a variável resposta de acordo com a regionalização escolhida, associando a barra de ocorrência da falta à sua região. O experimento utiliza apenas as 32 configurações de 2 a 33 regiões. Com 33 regiões, cada barra submetida às faltas constitui uma classe individual.

Não é necessário executar `Regionalizacoes.py` nem criar um dataset separado para cada regionalização. Os mesmos três datasets de atributos são reutilizados em todas as configurações.

## 7. Treinamento, otimização e avaliação — `Classificadores.py`

```powershell
python .\Classificadores.py
```

O script avalia Random Forest (RF), Regressão Logística (RL), Support Vector Machine (SVM), K-Nearest Neighbors (KNN) e XGBoost.

### 7.1. Protocolo experimental

```python
JANELAS = [(0, 5), (0, 10), (0, 20)]
NUMEROS_REGIOES = list(range(2, 34))
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_TRIALS = 100
N_FOLDS = 5
```

O particionamento 80/20 é estratificado pela barra e mantido para todas as janelas, regionalizações e classificadores. Para cada combinação de classificador, janela e número de regiões, a otimização utiliza Optuna com TPE e maximiza o Macro-F1 médio obtido por validação cruzada estratificada (`k-fold`, com `k = 5`) no treinamento, com estratificação pelas classes regionais.

O modelo com os melhores hiperparâmetros é treinado nas 1.096 simulações de treinamento e avaliado nas 274 simulações de teste, não utilizadas na seleção dos atributos ou no ajuste dos modelos. O escalonamento com `StandardScaler` faz parte dos pipelines de RL, SVM e KNN e é ajustado dentro de cada partição de treinamento. RF e XGBoost não utilizam esse escalonamento.

No total, são 480 configurações (5 classificadores × 3 janelas × 32 regionalizações). Com 100 tentativas por configuração, uma otimização completa sem reaproveitamento de caches pode exigir até 48.000 tentativas, cada uma com cinco partições.

### 7.2. Escolher os classificadores ativos

A lista `CLASSIFICADORES_ATIVOS` permite executar todos os modelos ou somente um subconjunto, por exemplo, para distribuir a otimização entre máquinas:

```python
CLASSIFICADORES_ATIVOS = [
    "Random Forest",
    "Regressão Logística",
    "SVM",
    "KNN",
    "XGBoost",
]
```

### 7.3. Reutilizar ou otimizar os hiperparâmetros

Os modos são definidos individualmente por classificador em `MODO_HIPERPARAMETROS`:

| Modo | Comportamento |
|---|---|
| `"carregar"` | Exige registro existente e compatível; não realiza nova otimização. |
| `"otimizar"` | Realiza nova otimização, mesmo que já exista registro. |
| `"auto"` | Reutiliza registros compatíveis; otimiza e salva quando não existem ou são incompatíveis. |

Exemplo para reutilizar os hiperparâmetros de todos os classificadores já otimizados:

```python
MODO_HIPERPARAMETROS = {
    "Random Forest": "carregar",
    "Regressão Logística": "carregar",
    "SVM": "carregar",
    "KNN": "carregar",
    "XGBoost": "carregar",
}
```

Os arquivos são armazenados em:

```text
Hiperparametros/
├── hiperparametros_rf_16.json
├── hiperparametros_rl_16.json
├── hiperparametros_svm_16.json
├── hiperparametros_knn_16.json
└── hiperparametros_xgboost_16.json
```

Cada arquivo contém registros por combinação de janela e regionalização (por exemplo, `0/5|8`), com hiperparâmetros, Macro-F1 de validação e assinatura de compatibilidade. Os caches são atualizados durante a otimização. Caches legados de RF e RL podem ser copiados automaticamente para os nomes/pastas atuais quando o arquivo de destino não existe.

Mantenha `VALIDAR_ASSINATURA_CACHE = True`. A assinatura considera dados de treinamento, configurações e versões das bibliotecas, conforme o modelo. Se houver incompatibilidade ao transferir JSONs entre máquinas, confira o ambiente e o protocolo antes de reutilizá-los. Não desative a validação apenas para contornar o erro. `CRIAR_BACKUP_AO_REOTIMIZAR = True` preserva uma cópia dos caches anteriores quando o modo `"otimizar"` é escolhido.

### 7.4. Resultados gerados

A versão compartilhada imprime, no terminal, as métricas de teste e as tabelas comparativas por janela, regionalização e classificador. Embora o artigo utilize exclusivamente o **Macro-F1**, o código também calcula a acurácia como métrica complementar.

Os gráficos são salvos em formato PDF vetorial:

```text
Graficos/
├── macro_f1.pdf
└── acuracia.pdf
```

Cada figura apresenta um painel por janela temporal, com a evolução da métrica em função do número de regiões. O PDF `macro_f1.pdf` é a figura utilizada no artigo.

Os hiperparâmetros são persistidos em JSON. A versão compartilhada não exporta automaticamente as tabelas completas de métricas para CSV ou JSON, nem serializa os modelos treinados. Para reconstruir os resultados de teste e os gráficos, execute novamente `Classificadores.py` com os caches compatíveis e os classificadores desejados.

## 8. Módulos auxiliares

Os módulos abaixo são importados pelos scripts principais e não devem ser executados isoladamente como etapas do pipeline.

| Módulo | Responsabilidade |
|---|---|
| `f_readpl4.py` | Leitura dos sinais armazenados no arquivo PL4 do ATP |
| `f_processar_pl4.py` | Seleção, recorte, reamostragem e alinhamento temporal dos sinais |
| `f_recortar_janela.py` | Recorte da janela relativa ao TTOQ |
| `f_selecionar_sinais_ponto.py` | Seleção das medições de um ponto, como `SUB` |
| `f_selecionar_sinais_medidores.py` | Seleção de sinais dos medidores, conforme a rotina que o utiliza |
| `f_preparar_sinais.py` | Preparação da soma das fases e dos sinais da fase faltosa |
| `f_extrair_harmonicas.py` | Extração das componentes harmônicas por FFT ciclo a ciclo |
| `f_calcular_parametros.py` | Cálculo dos parâmetros elétricos a partir das harmônicas |
| `f_calcular_estatisticas.py` | Cálculo das seis estatísticas de cada parâmetro ao longo da janela |

## 9. Problemas frequentes

| Problema | Verificação / solução |
|---|---|
| `python` não reconhecido | Confira a instalação e o PATH; no Windows, teste `py --version`. |
| `ModuleNotFoundError` para uma biblioteca externa | Ative a `.venv` correta e instale a dependência com `python -m pip install <biblioteca>`. |
| `ModuleNotFoundError` para `f_*.py` ou `Regionalizacoes` | Confira a estrutura do repositório e a presença dos arquivos locais; não são pacotes instaláveis via pip. |
| `runATP.exe` não encontrado | Verifique `RUNATP_PATH` e a instalação do ATP; não é necessário para as etapas posteriores se os Parquets existem. |
| Nenhum cartão P1/C1 encontrado | Confira `DIRETORIO_CARTOES`, a presença dos `.atp` diretamente na pasta configurada e o padrão `_p1_carregamento1_`. |
| Quantidade inesperada de linhas ou TTOQ fora de `t = 0` | Confira a taxa produzida por `f_processar_pl4.py` e a constante `AMOSTRAS_POR_CICLO` do gerador. |
| Parquets ausentes ou inválidos | Confira `Parquets/` ou `Parquets16/`, conforme a etapa. Um arquivo já existente é ignorado pelo gerador; se estiver inválido, isole-o antes de regenerar. |
| Erro ao identificar fase/barra | Preserve os padrões `FaseA`/`FaseB`/`FaseC` e `barraNN` nos nomes dos arquivos. |
| Atributos ausentes ou dados incompatíveis entre janelas | Confira os três datasets, as colunas de metadados, a amostragem e a ordem harmônica. Não misture as versões dos datasets. |
| Barra sem região | Confira se `REGIONALIZACOES` cobre todas as barras 2 a 34 sem duplicatas na mesma configuração. |
| Assinatura incompatível no cache | Verifique dados, parâmetros e versões das bibliotecas; utilize `auto` para reotimizar se necessário. |
| Gráficos incompletos | Confira `CLASSIFICADORES_ATIVOS`, `JANELAS` e `NUMEROS_REGIOES`; os gráficos refletem somente as configurações executadas. |

## 10. Checklist de reprodução

- [ ] Ambiente virtual configurado e dependências instaladas.
- [ ] Cartões P1/C1 disponíveis (se for necessário gerar os sinais).
- [ ] `Parquets/` e `DatasetAtributos/` completos, se for reproduzir a análise exploratória.
- [ ] Rankings ANOVA e RF obtidos sobre o conjunto de treinamento, utilizando a versão inicial dos atributos.
- [ ] `Parquets16/` contém os sinais de 1.370 simulações, com 640 linhas e 165 colunas por arquivo.
- [ ] `DatasetAtributos16/` contém os três datasets esperados, com 1.370 linhas e 111 colunas cada.
- [ ] `Regionalizacoes.py` contém as 32 regionalizações utilizadas (2 a 33 regiões).
- [ ] Seis atributos de interesse e protocolo experimental conferidos em `Classificadores.py`.
- [ ] Modos de hiperparâmetros definidos de acordo com a necessidade de otimização ou reaproveitamento dos JSONs.
- [ ] Classificadores executados e resultados de Macro-F1 conferidos no terminal e em `Graficos/macro_f1.pdf`.

---

**Nota de reprodução:** a ordem acima documenta o fluxo lógico do estudo. Os mesmos arquivos Python são reutilizados nas duas etapas; alteram-se somente as configurações necessárias, sem renomear ou duplicar scripts. Confira especialmente a taxa de reamostragem no módulo compartilhado `f_processar_pl4.py` e o diretório de entrada de `SelecaoAtributos.py`. As dimensões indicadas correspondem à base completa do experimento; a documentação não substitui a validação dos arquivos locais.

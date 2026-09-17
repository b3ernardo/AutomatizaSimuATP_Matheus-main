import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, hilbert
import pywt

#########################################################################################
# Energy
#########################################################################################

def df_energy(df, w):
    '''
    Calculates energy of signals in df for each window in list "w".
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i]
            energy = df[col].pow(2).groupby(df.index // window_size).mean()
            results.append(energy)
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

def df_energy_rolling(df, w):
    '''
    Calculates energy of signals in df for each window in list "w" (rolling window).
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i]
            energy = df[col].pow(2).rolling(window=window_size, step=window_size//2).mean()
            results.append(energy)
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

#########################################################################################
# Entropy
#########################################################################################

def df_entropy(df, w):
    '''
    Calculates Shannon entropy of signals in df for each window in list "w".
    Uses histogram 'fd' bins to estimate probability distribution.
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i]
            def get_shannon_entropy(series): # Internal function to calculate entropy of a series (window)
                data = series.dropna() # Remove NaNs to not break histogram
                if len(data) == 0:
                    return np.nan
                
                # Generates histogram to estimate probability density (p(x)) ‘fd’ (Freedman Diaconis) is a robust estimator for the number of bins
                counts, _ = np.histogram(data, bins='fd') # scipy.stats.entropy automatically normalizes counts to sum to 1
                return entropy(counts)
            
            # .apply() is necessary for 'entropy' isn't a native function of groupby
            signal_entropy = df[col].groupby(df.index // window_size).apply(get_shannon_entropy)
            results.append(signal_entropy)

        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

def df_entropy_rolling(df, w):
    '''
    Calculates Shannon entropy of signals in df for each window in list "w" (rolling window).
    Uses histogram 'fd' bins to estimate probability distribution.
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i]
            def get_shannon_entropy(series): # Internal function to calculate entropy of a series (window)
                data = series.dropna() # Remove NaNs to not break histogram
                if len(data) == 0:
                    return np.nan
                
                # Generates histogram to estimate probability density (p(x)) ‘fd’ (Freedman Diaconis) is a robust estimator for the number of bins
                counts, _ = np.histogram(data, bins='fd') # scipy.stats.entropy automatically normalizes counts to sum to 1
                return entropy(counts)
            
            # .apply() is necessary for 'entropy' isn't a native function of groupby
            signal_entropy = df[col].rolling(window=window_size, step=window_size//2).apply(get_shannon_entropy)
            results.append(signal_entropy)

        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

#########################################################################################
# Crest Factor
#########################################################################################

def df_crest_factor(df, w):
    '''
    Calculates Crest Factor (Peak / RMS) of signals in df for each window in list "w".
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i] 
            grouper = df.index // window_size
            peak_val = df[col].abs().groupby(grouper).max()
            rms_val = df[col].pow(2).groupby(grouper).mean().pow(0.5)
            crest_factor = peak_val / rms_val
            results.append(crest_factor)
            
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

def df_crest_factor_rolling(df, w):
    '''
    Calculates Crest Factor (Peak / RMS) of signals in df for each window in list "w" (rolling window).
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i] 
            grouper = df.index // window_size
            peak_val = df[col].abs().rolling(window=window_size, step=window_size//2).max()
            rms_val = df[col].pow(2).rolling(window=window_size, step=window_size//2).mean().pow(0.5)
            crest_factor = peak_val / rms_val
            results.append(crest_factor)
            
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

#########################################################################################
# Kurtosis
#########################################################################################

def df_kurtosis(df, w):
    '''
    Calculates Kurtosis of signals in df for each window in list "w".
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i] 
            kurtosis_val = df[col].groupby(df.index // window_size).kurt()
            results.append(kurtosis_val)
            
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

def df_kurtosis_rolling(df, w):
    '''
    Calculates Kurtosis of signals in df for each window in list "w".
    Filters out mathematically impossible values ( < -2).
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i] 
            kurtosis_val = df[col].rolling(window=window_size, step=window_size//2).kurt()
            
            # Substitui valores menores que -2 por NaN
            # Isso elimina o "lixo" de -3.0 sem deletar o timestamp correspondente
            
            kurtosis_val = kurtosis_val.mask(kurtosis_val < -2, np.nan)
            
            results.append(kurtosis_val)
            
        except (IndexError, KeyError):
            print(f"Warning: There is no window for column '{col}'.")
    
    return pd.concat(results, axis=1)

#########################################################################################
# Wavelet Transform
#########################################################################################
# Verificar, por enquanto muito pesada computacionalmente

def df_wavelet_transform(df, wavelet, level):
    '''
    Calculates Discrete Wavelet Transform (DWT) of signals in df using specified wavelet and level.
    Returns a list of DataFrames with wavelet coefficients for each signal.
    '''
    wavelet_results = []

    for col in df.columns:
        coeffs = pywt.wavedec(df[col].values, wavelet, level=level)
        coeffs_df = pd.DataFrame(coeffs).transpose()
        wavelet_results.append(coeffs_df)

    return wavelet_results

def df_wavelet_transform_rolling(df, wavelet, level, window_size):
    '''
    Calculates Discrete Wavelet Transform (DWT) of signals in df using specified wavelet and level (rolling window).
    Returns a list of DataFrames with wavelet coefficients for each signal.
    '''
    wavelet_results = []
    for col in df.columns:
        def compute_dwt(series):
            coeffs = pywt.wavedec(series.values, wavelet, level=level)
            return pd.Series(coeffs)
        
        coeffs_df = df[col].rolling(window=window_size, step=window_size//2).apply(compute_dwt, raw=False)
        wavelet_results.append(coeffs_df)
    return wavelet_results

#########################################################################################
# FFT
#########################################################################################

def df_fft(df, fs):
    '''
    Calculates the FFT of the signal for each column of df with it's sampling frequency.
    Returns a DataFrame where:
      - Index (rows): Harmonic order (0 to 30)
      - Columns: Original column names
      - Values: Magnitude in PU (Per Unit), normalized by the fundamental.
    '''
    results = {} # Dicionário para armazenar as Séries

    for i, col in enumerate(df.columns):
        try:
            # 1. Preparação do Sinal (Remove NaNs e pega o sinal completo)
            sinal = df[col].dropna().values
            N = len(sinal)
            fsi = fs[i]
            
            if N == 0:
                print(f"Warning: Column '{col}' is empty.")
                continue

            # 2. Cálculo da FFT (rfft é mais rápido para sinais reais)
            yf = rfft(sinal)
            xf = rfftfreq(N, 1/fsi)
            
            # Magnitude normalizada (2/N para AC, 1/N para DC, simplificando para 2/N geral)
            magnitude = (2.0 / N) * np.abs(yf)

            # 3. Identifica a Fundamental (Busca pico na região de 50Hz a 70Hz)
            # Isso é necessário para definir quem é o "1.0 PU"
            mask_search = (xf >= 40) & (xf <= 80)
            if mask_search.any():
                idx_region = np.where(mask_search)[0]
                # Pega o índice do pico máximo dentro dessa região
                idx_peak = idx_region[np.argmax(magnitude[idx_region])]
                f0 = xf[idx_peak]     # Frequência fundamental exata
                amp_f0 = magnitude[idx_peak] # Amplitude de referência
            else:
                # Fallback caso não ache nada (sinal DC puro ou ruído)
                f0 = 60.0
                amp_f0 = 1.0 

            # 4. Extração das Harmônicas (0 a 30)
            harmonics_vals = []
            harmonics_idx = range(31) # 0 a 30

            for h in harmonics_idx:
                target_freq = f0 * h
                
                # Encontra o índice da frequência mais próxima no vetor da FFT
                idx_closest = np.argmin(np.abs(xf - target_freq))
                
                # Pega amplitude e calcula PU
                amp_abs = magnitude[idx_closest]
                val_pu = amp_abs / amp_f0 if amp_f0 > 0 else 0.0
                
                harmonics_vals.append(val_pu)

            # Salva na estrutura (cria uma Série com índices 0..30)
            results[col] = pd.Series(harmonics_vals, index=harmonics_idx)
            
        except Exception as e:
            print(f"Error processing column '{col}': {e}")
    
    # Concatena todas as séries em um único DataFrame (mesma estrutura de retorno)
    return pd.DataFrame(results)

#########################################################################################
# Harmonic extraction
#########################################################################################

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def df_bandpass_filtering(df, fs, f_center, bandwidth=10.0, order=4):
    '''
    Applies a Butterworth Bandpass filter to each column in df.
    The filter is applied to the full signal to avoid windowing artifacts.
    
    Parameters:
    - df: Input DataFrame with raw signals.
    - fs: Sampling frequency in Hz.
    - f_center: Target frequency (e.g., 60, 120, 180 Hz).
    - bandwidth: Width of the passband in Hz.
    - order: Filter order (higher means steeper roll-off).
    '''
    results = []

    for i,col in enumerate(df.columns):
        try:
            # 1. Design the filter coefficients (Nyquist requirement: $f_{nyq} = \frac{fs}{2}$)
            nyq = 0.5 * fs[i]
            low = (f_center - bandwidth / 2) / nyq
            high = (f_center + bandwidth / 2) / nyq
            
            # Constrain frequencies to valid range (0 to 1)
            if low <= 0: low = 0.001
            if high >= 1: high = 0.999

            b, a = butter(order, [low, high], btype='band')
            # 2. Extract raw values and handle NaNs
            # We drop NaNs to filter, then we will re-align if necessary.
            # For power signals, continuity is key.
            raw_data = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
            
            if len(raw_data) < (order * 3):
                # filtfilt requires a minimum signal length to be stable
                filtered_series = pd.Series([np.nan] * len(df[col]), index=df.index)
            else:
                # 3. Apply Zero-Phase Filtering (filtfilt)
                # This ensures the filtered harmonic is perfectly aligned with the original wave
                y = filtfilt(b, a, raw_data)
                
                # Reconstruct as a Series to preserve index
                filtered_series = pd.Series(y, name=f"f{int(f_center)}_{col}")
            
            results.append(filtered_series)

        except Exception as e:
            print(f"Warning: Could not process column '{col}'. Error: {e}")
    
    # 4. Concatenate results into a new DataFrame
    return pd.concat(results, axis=1)

def df_instantaneous_phase(df):
    '''
    Extracts the instantaneous phase angle (degrees) of signals in df using the Hilbert Transform.
    
    Parameters:
    - df: Input DataFrame (should be pre-filtered for a specific harmonic/frequency).
    '''
    results = []

    for col in df.columns:
        try:
            # 1. Limpeza e preparação (ignora NaNs/Infs para não quebrar a Transformada)
            raw_data = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
            
            if len(raw_data) == 0:
                # Retorna NaNs se a coluna estiver vazia
                phase_series = pd.Series([np.nan] * len(df[col]), index=df.index)
            else:
                # 2. Sinal Analítico: z(t) = s(t) + j*H[s(t)]
                # Onde H[s(t)] é a Transformada de Hilbert (componente em quadratura)
                analytic_signal = hilbert(raw_data)
                
                # 3. Extração do Ângulo em Radianos [-pi, pi]
                phase_rad = np.angle(analytic_signal)
                
                # 4. Conversão para Graus [-180, 180]
                phase_deg = np.degrees(phase_rad)
                
                # Reconstrói a série mantendo o nome original com sufixo
                phase_series = pd.Series(phase_deg, name=f"phase_{col}")
            
            results.append(phase_series)

        except Exception as e:
            print(f"Warning: Could not process phase for column '{col}'. Error: {e}")
    
    # Retorna o DataFrame com as fases alinhadas pela primeira linha
    return pd.concat(results, axis=1)

def df_relative_phase_shift(df_phase_fund, df_phase_harm, harmonic_order):
    '''
    Calculates the relative phase shift between the fundamental and a specific harmonic.
    Equation: $\Delta\phi = (\phi_{harm} - (n \cdot \phi_{fund})) \pmod{360}$
    
    Parameters:
    - df_phase_fund: DataFrame containing the instantaneous phase of the fundamental (60Hz).
    - df_phase_harm: DataFrame containing the instantaneous phase of the harmonic (e.g., 180Hz).
    - harmonic_order: The integer order of the harmonic (3 for 3rd, 5 for 5th, etc.).
    '''
    results = []

    # Iteramos pelas colunas. Assume-se que ambos os DFs possuem a mesma estrutura.
    for i,col in enumerate(df_phase_fund.columns):
        try:
            col_fund = df_phase_fund.columns[i]
            col_harm = df_phase_harm.columns[i]
            
            # 1. Extração dos dados e alinhamento
            # Usamos o índice do df_phase_fund como referência
            phase_f = df_phase_fund[col_fund].values
            phase_h = df_phase_harm[col_harm].values
            
            # 2. Cálculo da Defasagem Relativa
            # O multiplicador (harmonic_order) sincroniza as velocidades angulares
            relative_phase = (phase_h - (harmonic_order * phase_f)) % 360
            
            # 3. Reconstrução da Série
            # Nomeamos para indicar a relação (Ex: Signal1_h3_rel_phase)
            series_name = f"h{harmonic_order}_rel_{col}"
            res_series = pd.Series(relative_phase, name=series_name, index=df_phase_fund.index)
            
            results.append(res_series)

        except Exception as e:
            print(f"Warning: Could not calculate relative phase for index {i}. Error: {e}")
    
    # Retorna o DataFrame Multi-coluna alinhado
    return pd.concat(results, axis=1)

def df_phase_derivative(df):
    '''
    Calculates the rate of change (derivative) of the phase angle.
    CRITICAL: Handles the circular wrap-around (0-360 jump) using np.unwrap.
    FIX: Handles variable length due to NaNs by preserving the valid index.
    
    Returns:
    - derivative in degrees per sample ($d\phi/dt$).
    '''
    results = []

    for col in df.columns:
        try:
            # 1. Preparação dos dados mantendo o Index
            # Em vez de pegar só os .values, pegamos a Series inteira limpa
            valid_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            # Guardamos os valores E o índice onde eles ocorrem
            raw_data = valid_series.values
            valid_index = valid_series.index 

            if len(raw_data) == 0:
                # Se tudo for NaN, retorna uma série de NaNs com o índice original completo
                deriv_series = pd.Series(np.nan, index=df.index, name=f"derivative_{col}")
            else:
                # 2. Conversão e Unwrap
                rads = np.deg2rad(raw_data)
                unwrapped_rads = np.unwrap(rads)
                
                # 3. Cálculo do Gradiente
                deriv_rads = np.gradient(unwrapped_rads)
                
                # 4. Retorno para Graus
                deriv_deg = np.rad2deg(deriv_rads)
                
                # 5. Reconstrução da Série usando o ÍNDICE VÁLIDO (A Correção)
                # Agora o comprimento de deriv_deg (16622) bate com valid_index (16622)
                deriv_series = pd.Series(deriv_deg, name=f"derivative_{col}", index=valid_index)
                
                # Opcional: Reindexar para garantir que volte ao tamanho original (94452) preenchendo o resto com NaN
                deriv_series = deriv_series.reindex(df.index)
            
            results.append(deriv_series.iloc[300:-300]) # Remove os 300 primeiros e 300 últimos para evitar bordas instáveis

        except Exception as e:
            print(f"Warning: Could not calculate derivative for column '{col}'. Error: {e}")
            # Em caso de erro, adiciona uma coluna vazia para não quebrar a estrutura do DF final
            results.append(pd.Series(np.nan, index=df.index, name=f"derivative_{col}"))
    
    # O pd.concat alinha automaticamente pelos índices
    return pd.concat(results, axis=1)

#########################################################################################
# RMS and amplitude analysis
#########################################################################################

def df_rms_rolling(df, w):
    '''
    Calculates the Rolling RMS (Root Mean Square). 
    Can input a "Ripple" effect due to fixed windows, if the signal frequency isn't perfectly equal to nominal freq.
    
    Parameters:
    - w: Number of samples in the window (e.g., samples per cycle).
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            window_size = w[i]  # Use o tamanho correspondente ou o último se faltar
            # Cálculo vetorizado eficiente do RMS móvel
            # RMS = sqrt(mean(x^2))
            rms_series = df[col].pow(2).rolling(window=window_size).mean().apply(np.sqrt)
            
            # Renomeia para manter organização
            rms_series.name = f"rms_{col}"
            results.append(rms_series)

        except Exception as e:
            print(f"Warning: Could not process RMS for '{col}'. Error: {e}")
            
    return pd.concat(results, axis=1)

def df_envelope_rms(df):
    '''
    Calculates the RMS Amplitude using the Hilbert Transform Envelope.
    FIX: Handles variable length due to NaNs by preserving the valid index.
    
    Returns:
    - RMS value (Amplitude / sqrt(2)) aligned with original index.
    '''
    results = []

    for col in df.columns:
        try:
            # 1. Limpeza Inteligente (Guardando o Índice)
            # Removemos NaNs, mas guardamos QUEM são os dados válidos
            valid_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            raw_data = valid_series.values
            valid_index = valid_series.index  # <--- A CHAVE DO PROBLEMA
            
            if len(raw_data) == 0:
                # Se a coluna for toda vazia, retorna série de NaNs do tamanho original
                rms_series = pd.Series(np.nan, index=df.index, name=f"rms_{col}")
            else:
                # 2. Envelope Analítico (Hilbert)
                analytic_signal = hilbert(raw_data)
                amplitude_envelope = np.abs(analytic_signal)
                
                # 3. Conversão para RMS (Pico / sqrt(2))
                rms_values = amplitude_envelope / np.sqrt(2)
                
                # 4. Reconstrução usando o ÍNDICE VÁLIDO
                # Agora o tamanho de rms_values bate com valid_index
                rms_series = pd.Series(rms_values, name=f"rms_{col}", index=valid_index)
                
                # 5. Reindexação final
                # Expande de volta para o tamanho total (94452), colocando NaNs onde não calculamos
                rms_series = rms_series.reindex(df.index)
            
            results.append(rms_series)

        except Exception as e:
            print(f"Error calculating Envelope RMS for '{col}': {e}")
            # Em caso de erro fatal, garante que o código não pare, retornando vazio
            results.append(pd.Series(np.nan, index=df.index, name=f"rms_{col}"))
            
    return pd.concat(results, axis=1)

def df_temporal_derivative_simple(df, t):
    '''
    Calculates the Temporal Derivative using Backward Difference (Simple Difference).
    Equation: dx/dt = (x[t] - x[t-1]) / (1/t)

    Parameters:
    - t: Window size in samples, ususally (sample rate/fundamental frequency).
    
    CRITICAL NOTE: This method introduces a phase shift of 0.5 samples.
    It represents the derivative at time t - (dt/2), not exactly at t.
    '''
    results = []

    for i, col in enumerate(df.columns):
        try:
            dt = 1.0 / t[i] # Intervalo de tempo entre amostras
            # 1. Extração dos dados
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(series) < 2:
                # Precisa de pelo menos 2 pontos para subtrair
                deriv_series = pd.Series(np.nan, index=df.index, name=f"deriv_simple_{col}")
            else:
                # 2. Cálculo da Diferença (Diff)
                # diff() calcula x[t] - x[t-1]. O primeiro elemento vira NaN.
                delta_x = series.diff(periods=t[i])
                
                # 3. Divisão pelo Delta T (para virar Taxa de Variação Física)
                # Ex: Se x é Amperes, o resultado é Amperes/segundo
                deriv_values = delta_x / dt
                
                # Opcional: Preencher o primeiro NaN com 0 ou com o segundo valor (bfill)
                # Para manter pureza estatística, deixamos NaN.
                
                deriv_series = deriv_values
                deriv_series.name = f"deriv_simple_{col}"
                
                # Reindexa para garantir alinhamento com o DF original
                deriv_series = deriv_series.reindex(df.index)

            results.append(deriv_series) # .abs() para evitar valores negativos (opcional)

        except Exception as e:
            print(f"Error calculating simple derivative for {col}: {e}")

    return pd.concat(results, axis=1)


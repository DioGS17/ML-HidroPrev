import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


def data_processing(path: str, date_start: str, date_end: str, arquivos=None) -> pd.DataFrame:
    # Guardando os nomes dos arquivos
    if arquivos is None:
        arquivos = [x for x in os.listdir(path) if x.endswith('.csv')]

    # Abrindo CSVs
    df_cotas = []
    for i in arquivos:
        df_cotas.append(pd.read_csv(f'{path}/{i}', sep = ';'))

    for df in df_cotas:
        # Removendo dados duplicados
        df.drop_duplicates(inplace=True)
        df.drop_duplicates(subset=['DataCompleta'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Convertendo a coluna 'DataCompleta' para datetime e definindo como índice
        df['DataCompleta'] = pd.to_datetime(df['DataCompleta'])
        df.index = df['DataCompleta']
        df.drop(columns=['DataCompleta'], inplace=True)

        # Transformando a coluna cota no tipo float64
        if df['Cota'].dtype == np.dtype('O'):
            df['Cota'] = df['Cota'].str.replace(',', '.', regex=False).astype('float64')

    datas = pd.date_range(start=date_start, end=date_end, freq='D')
    datas = pd.DataFrame(datas, columns=['DataCompleta'])
    datas.set_index('DataCompleta', inplace=True)

    for i in range(len(arquivos)):
        datas[arquivos[i][22:-4]] = df_cotas[i]['Cota'].loc[date_start:date_end]

    # Interpolação dos dados
    dados = datas.copy()
    dados.interpolate(inplace=True)

    dados.index.name = 'Data'
    
    return dados


def seasonal_plot(df: pd.DataFrame, figsize: tuple[int, int], with_std=True):
    fig, ax = plt.subplots(df.shape[1], 1, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=1)
    ax = ax.ravel()

    for i, col in enumerate(df):
        mean = df[col].groupby(df.index.strftime('%m%d')).mean()

        ax[i].plot(mean, label='Média', color='blue')
        if with_std:
            std = np.std(df[col])
            ax[i].fill_between(mean.index, mean - std, mean + std, alpha=0.2, label='Desvio Padrão')
        ax[i].set_title(f'Cota {col}')
        ax[i].set_xlabel('Mês')
        ax[i].set_ylabel('Cota')
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[i].legend(loc='upper right')
        ax[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Variação Sazonal das Cotas', fontsize=16, y=1.02)
    plt.show()


def _max_min_year_(series):
  max_year = []
  min_year = []
  first_year =  series.index[0].year
  last_year = series.index[-1].year

  for i in range(first_year,last_year+1):
    aux = series[str(i)+'-01-01':str(i)+'-12-31']
    max_year.append(aux.idxmax())
    min_year.append(aux.idxmin())

  return max_year,min_year


def annual_plot(df: pd.DataFrame, figsize: tuple[int, int]):
    fig, ax = plt.subplots(df.shape[1], 1, figsize=figsize, sharey=True)
    fig.subplots_adjust(hspace=1)
    ax = ax.ravel()

    for i, col in enumerate(df):
        max_year, min_year = _max_min_year_(df[col])
        
        ax[i].plot(df[col], label='Cota')
        ax[i].scatter(max_year, df[col].loc[max_year], marker='x', color='red', label='Máximo Anual')
        ax[i].scatter(min_year, df[col].loc[min_year], marker='o', color='green', label='Mínimo Anual')
        
        ax[i].set_title(f'Cota {col}')
        ax[i].set_xlabel('Ano')
        ax[i].set_ylabel('Cota')
        ax[i].legend()
        ax[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Variação Anual das Cotas', fontsize=16, y=1.02)
    plt.show()


def cross_corr(df: pd.DataFrame, tgt: str):
    dados_dessazonalizados = df.copy()

    for c in df.columns:
        decompostion = seasonal_decompose(df[c], model='additive', period=365)
        seasonal = decompostion.seasonal
        dados_dessazonalizados[c] = df[c] - seasonal

    for c in dados_dessazonalizados.columns:
        if c == tgt:
            continue
        best_lag = None
        max_corr = -np.inf

        for i in range(32):
            corr = dados_dessazonalizados[tgt].corr(dados_dessazonalizados[c].shift(i), method='spearman')

            if corr > max_corr:
                max_corr = corr
                best_lag = i
        
        print(f"Estação {c}: Melhor lag = {best_lag}; Correlação = {max_corr:.4f}")
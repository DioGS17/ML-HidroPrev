import os
import pandas as pd
from calendar import monthrange

def criar_data_completa(data_base, dia):
    """Cria uma data completa, garantindo que o dia seja válido para o mês."""
    if pd.isna(data_base) or pd.isna(dia):
        return pd.NaT  # Retorna uma data vazia se houver valores nulos
    ano = data_base.year
    mes = data_base.month
    if dia > monthrange(ano, mes)[1]:  # Verifica se o dia é válido para o mês
        return pd.NaT
    return data_base.replace(day=dia)


def processar_dados(file_path):
    nome_arquivo = os.path.basename(file_path)
    numero_estacao = nome_arquivo.split('_')[0]

    # Abrindo CSV
    df = pd.read_csv(file_path, delimiter=';', encoding='latin1', skiprows=15)

    # Separando as colunas Data e Cota01-31
    cotas =  [f'Cota{i:02d}' for i in range(1, 32)]
    colunas = ['Data'] + cotas
    dados = df[colunas]

    # Removendo duplicatas
    dados.drop_duplicates(inplace=True)

    # Transformando a coluna Data no tipo datetime
    dados['Data'] = pd.to_datetime(dados['Data'], format='%d/%m/%Y', errors='coerce')

    # Transformando as colunas de cotas em formato longo
    dados_long = pd.melt(dados, id_vars=['Data'], value_vars=cotas, var_name='Dia', value_name='Cota')

    # Extrai o número do dia
    dados_long['Dia'] = dados_long['Dia'].str.extract(r'(\d+)').astype(int)

    # Cria a coluna DataCompleta com a função
    dados_long['DataCompleta'] = dados_long.apply(lambda row: criar_data_completa(row['Data'], row['Dia']), axis=1)

    # Remover dados nulos e colunas Data e Dia, e ordernar pela coluna DataCompleta
    dados_diarios = dados_long[['DataCompleta', 'Cota']].dropna().sort_values(by='DataCompleta')
    dados_diarios.reset_index(inplace=True, drop=True)

    # Alguns dias possui mais de uma cota, então agrupa as cotas do mesmo dia e faz a média
    dados_diarios_2 = dados_diarios.groupby(['DataCompleta']).mean().round(0)

    # Coloque o caminho para salvar o CSV
    output_csv_path = os.path.join("/home/dsantos/ML HidroPrev/Datasets processados", f'dados_diarios_estacao_{numero_estacao}.csv')

    # Salvando CSV
    dados_diarios_2.to_csv(output_csv_path, sep=';', encoding='utf-8')


def selecionar_arquivo():
    return input("Digite o caminho do arquivo CSV: ")


if __name__=='__main__':
    arquivo_csv = selecionar_arquivo()
    processar_dados(arquivo_csv)

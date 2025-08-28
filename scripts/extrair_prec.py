import pandas as pd

def extraction(path: str, name: str):
    prec = pd.read_csv(path,
                    sep=r'\s+',
                    comment='#',
                    names=['Data', 'Lat', 'Lon', 'value'])
    
    prec['Data'] = pd.to_datetime(prec['Data'])
    prec.index = prec['Data']
    prec.drop(columns=['Data','Lat', 'Lon'], inplace=True)

    prec.to_csv(f'Precipitação/{name}_prec.csv', sep=';')


def selecionar_arquivo():
    return input("Digite o caminho do arquivo: ")


if __name__ == '__main__':
    csv = selecionar_arquivo()
    name = input("Digite o nome para salvar: ")
    extraction(csv, name)
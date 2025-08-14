#!/bin/bash

# Define o URL base
BASE_URL="https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/sst/"

# Define o diretório de destino onde os arquivos serão salvos
DOWNLOAD_DIR="dados_sst_netcdf"

# Cria o diretório de destino se ele não existir
mkdir -p "$DOWNLOAD_DIR"

echo "Iniciando o download dos arquivos NetCDF..."
echo "Arquivos existentes não serão baixados novamente."

# Loop pelos anos de 1985 a 2025
for year in $(seq 1985 2025); do
    YEAR_URL="${BASE_URL}${year}/"
    echo "Baixando dados do ano: $year"

    # Baixa todos os arquivos .nc (NetCDF) do URL do ano para o diretório de destino
    # -r: recursivo (não estritamente necessário aqui, mas boa prática para wget)
    # -np: não subir para o diretório pai
    # -nd: não criar diretórios hierárquicos (baixa tudo para DOWNLOAD_DIR)
    # -A ".nc": aceita apenas arquivos com a extensão .nc
    # -q: modo silencioso (menos saída no terminal)
    # -nc: não sobrescreve arquivos existentes (no-clobber)
    wget -r -np -nd -A ".nc" -P "$DOWNLOAD_DIR" -nc "$YEAR_URL"
done

echo "Download concluído! Todos os arquivos NetCDF foram salvos em: $DOWNLOAD_DIR"

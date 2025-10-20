from pathlib import Path
from src.preprocessing import load_and_clean_data, create_temporal_features

csv_path = Path("raw/merged_dataset.csv") 
geocode_cache_path = Path("raw/geocode_cache.json")


DROP_COLUMNS = [
    "detalhe_endereco_acidente", "numero_cruzamento", "num_semaforo",
    "sentido_via", "acidente_verificado", "tempo_clima", "situacao_semaforo",
    "sinalizacao", "condicao_via", "conservacao_via", "ponto_controle",
    "situacao_placa", "velocidade_max_via", "mao_direcao", "divisao_via1",
    "divisao_via2", "divisao_via3", "Protocolo"
]

CATEGORICAL_COLUMNS = [
    "natureza_acidente", "situacao", "bairro", "endereco",
    "numero", "complemento", "endereco_cruzamento",
    "referencia_cruzamento", "bairro_cruzamento", "tipo", "descricao"
]
NUMERIC_COLUMNS = [
    "auto", "moto", "ciclom", "ciclista", "pedestre",
    "onibus", "caminhao", "viatura", "outros",
    "vitimas", "vitimasfatais"
]


df = load_and_clean_data(
    csv_path,
    geocode_cache_path,
    DROP_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS
)

print(df.shape)
print(df.head())

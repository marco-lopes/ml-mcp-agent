# Guia de Uso do Kaggle com kagglehub

Este documento explica como usar a integração com Kaggle através da biblioteca `kagglehub`.

## Instalação

```bash
pip install kagglehub
```

## Configuração de Credenciais

### Método 1: Token OAuth (Recomendado)

Configure a variável de ambiente `KAGGLE_KEY`:

```bash
export KAGGLE_KEY="KAGGLE_KEY_seu_token_aqui"
```

Ou crie um arquivo `.env` na raiz do projeto:

```bash
echo "KAGGLE_KEY=KAGGLE_KEY_seu_token_aqui" > .env
```

### Método 2: Arquivo kaggle.json (Legado)

1. Baixe suas credenciais em https://www.kaggle.com/account
2. Coloque o arquivo em `~/.kaggle/kaggle.json`
3. Ajuste as permissões: `chmod 600 ~/.kaggle/kaggle.json`

## Como Funciona o kagglehub

O `kagglehub` gerencia automaticamente um **cache local** de datasets:

- **Localização do cache**: `~/.cache/kagglehub/datasets/`
- **Estrutura**: `~/.cache/kagglehub/datasets/{owner}/{dataset}/versions/{version}/`
- **Vantagem**: Downloads subsequentes são instantâneos (usa o cache)

### Exemplo de Download

```python
import kagglehub

# Download do dataset Iris
path = kagglehub.dataset_download("uciml/iris")

# Retorna algo como:
# /home/user/.cache/kagglehub/datasets/uciml/iris/versions/2
```

## Uso no Servidor MCP

O servidor Kaggle (`servers/kaggle_server/main.py`) foi corrigido para usar `kagglehub` corretamente:

### Tool: `download_dataset`

```python
# Via MCP
result = kaggle_client.call_tool(
    "download_dataset",
    dataset_ref="uciml/iris"
)

# Retorna:
{
    "success": True,
    "dataset_ref": "uciml/iris",
    "output_dir": "/home/user/.cache/kagglehub/datasets/uciml/iris/versions/2",
    "files": [
        {
            "name": "Iris.csv",
            "path": "/home/user/.cache/kagglehub/datasets/uciml/iris/versions/2/Iris.csv",
            "size_bytes": 5107
        }
    ],
    "total_files": 1
}
```

### Tool: `search_datasets`

Usa o Kaggle CLI para busca (kagglehub não tem funcionalidade de busca):

```python
result = kaggle_client.call_tool(
    "search_datasets",
    query="iris",
    max_results=10
)
```

## Uso Direto (Sem MCP)

Para usar `kagglehub` diretamente em seus scripts:

```python
import os
import kagglehub
from pathlib import Path

# Configurar token
os.environ["KAGGLE_KEY"] = "seu_token_aqui"

# Download
dataset_path = kagglehub.dataset_download("uciml/iris")

# Listar arquivos
for file in Path(dataset_path).glob("*.csv"):
    print(f"Arquivo CSV: {file}")
    
# Usar em treinamento
import pandas as pd
df = pd.read_csv(Path(dataset_path) / "Iris.csv")
```

## Integração com Pipeline de ML

### Exemplo Completo

```python
import os
import kagglehub
from pathlib import Path
import sys

# Configurar token
os.environ["KAGGLE_KEY"] = "seu_token_aqui"

# 1. Download do dataset
print("Baixando dataset...")
dataset_path = kagglehub.dataset_download("uciml/iris")
csv_file = Path(dataset_path) / "Iris.csv"

# 2. Treinar modelo
sys.path.insert(0, "agent")
from main import MLAgent

agent = MLAgent()
agent.start()

result = agent.train_model(
    model_name="iris_classifier",
    model_type="random_forest",
    dataset_path=str(csv_file),
    target_column="Species",
    task_type="classification",
    hyperparameters={"n_estimators": 100}
)

print(f"Acurácia: {result['test_metrics']['val_accuracy']}")

agent.stop()
```

## Diferenças: kagglehub vs Kaggle CLI

| Recurso | kagglehub | Kaggle CLI |
|---------|-----------|------------|
| Download de datasets | ✅ Sim | ✅ Sim |
| Busca de datasets | ❌ Não | ✅ Sim |
| Cache automático | ✅ Sim | ❌ Não |
| API Python nativa | ✅ Sim | ❌ Não (subprocess) |
| Velocidade | ⚡ Rápido | 🐌 Mais lento |
| Gerenciamento de versões | ✅ Automático | ⚙️ Manual |

## Correções Implementadas

### Problema Original

O código tentava passar `output_dir` para `kagglehub.dataset_download()`, mas essa função não aceita esse parâmetro.

```python
# ❌ ERRADO
download_path = kagglehub.dataset_download(
    dataset_ref,
    output_dir=output_dir,  # Não existe!
    force_download=force_download,  # Não existe!
)
```

### Solução Implementada

```python
# ✅ CORRETO
download_path = kagglehub.dataset_download(dataset_ref)

# Se usuário quer output_dir específico, copiar arquivos
if output_dir:
    import shutil
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(download_path)
    for item in source_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
    
    download_path = str(output_path)
```

## Exemplos Disponíveis

### 1. Download Simples

```bash
python3 examples/kaggle_download_example.py
```

Demonstra:
- Download com kagglehub
- Listagem de arquivos
- Preview de dados
- Uso no pipeline

### 2. Pipeline Completo

```bash
python3 examples/real_training_example.py
```

Demonstra:
- Busca de datasets
- Download
- Treinamento com scikit-learn
- Deploy

## Troubleshooting

### Erro: "kagglehub not installed"

```bash
pip install kagglehub
```

### Erro: "Kaggle credentials not configured"

Verifique se o token está configurado:

```bash
echo $KAGGLE_KEY
```

Se não estiver, configure:

```bash
export KAGGLE_KEY="KAGGLE_KEY_seu_token_aqui"
```

### Erro: "Dataset not found"

Verifique se o dataset existe no Kaggle:

```bash
# Buscar datasets
kaggle datasets list -s "iris"
```

### Cache cheio

Limpar cache do kagglehub:

```bash
rm -rf ~/.cache/kagglehub
```

## Referências

- [kagglehub Documentation](https://github.com/Kaggle/kagglehub)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

## Conclusão

O sistema agora usa `kagglehub` corretamente para:

✅ Download eficiente de datasets  
✅ Cache automático  
✅ Integração com pipeline de ML  
✅ Compatibilidade com Kaggle CLI para buscas  

Para qualquer dúvida, consulte os exemplos em `examples/`.

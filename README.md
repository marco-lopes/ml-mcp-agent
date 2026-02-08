> Este projeto foi gerado pela IA Manus para demonstrar a integração de um agente de orquestração de Machine Learning com servidores MCP (Model Context Protocol).

# Agente de Orquestração de ML com MCP

Este repositório contém um sistema completo para orquestrar o treinamento e o deploy de modelos de Machine Learning. O sistema utiliza o **Model Context Protocol (MCP)** para a comunicação entre um agente central e três servidores especializados: um para gerenciar datasets do Kaggle (usando `kagglehub`), um para **treinamento real de modelos com scikit-learn**, e outro para deploy.

## Visão Geral da Arquitetura

O sistema é composto por quatro componentes principais que se comunicam via MCP sobre `stdio`:

1.  **Agente Orquestrador**: O cérebro do sistema. Ele gerencia o pipeline de ponta a ponta, desde a busca de um dataset no Kaggle até o deploy do modelo treinado.
2.  **Servidor de Datasets Kaggle**: Um novo servidor MCP que expõe `tools` para buscar, baixar e gerenciar datasets da plataforma Kaggle.
3.  **Servidor de Treinamento**: Um servidor MCP que expõe `tools` para treinar, validar e gerenciar modelos de ML **usando scikit-learn**. Ele lida com o treinamento real, validação e armazenamento de modelos e metadados.
4.  **Servidor de Deploy**: Um terceiro servidor MCP que oferece `tools` para fazer o deploy de modelos treinados, criar endpoints de inferência e gerenciar o ciclo de vida dos deployments.

Para uma descrição mais detalhada, consulte o documento de arquitetura: [architecture.md](architecture.md).

## Estrutura do Projeto

```
ml-mcp-agent/
├── README.md
├── requirements.txt
├── architecture.md
├── agent/
│   ├── main.py                 # Agente principal e ponto de entrada
│   ├── mcp_client.py           # Cliente genérico para comunicação com servidores MCP
│   └── orchestrator.py         # Lógica de orquestração do pipeline de ML
├── servers/
│   ├── kaggle_server/          # Servidor MCP para datasets Kaggle
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── training_server/        # Servidor MCP para treinamento com scikit-learn
│   │   ├── main.py
│   │   ├── models.py             # Implementação dos modelos com scikit-learn
│   │   ├── utils.py              # Utilitários de dados e métricas
│   │   └── requirements.txt
│   └── deployment_server/      # Servidor MCP para deploy
│       ├── main.py
│       └── requirements.txt
├── examples/
│   ├── train_and_deploy.py     # Script de exemplo original (simulado)
│   └── real_training_example.py # Exemplo com treinamento real (scikit-learn)
└── tests/
    └── (vazio)                 # Diretório para futuros testes
```

## Funcionalidades

- **Comunicação via MCP**: Utiliza o protocolo MCP para uma comunicação padronizada e desacoplada.
- **Integração com Kaggle**: Ferramentas para buscar e baixar datasets diretamente do Kaggle usando a biblioteca `kagglehub`.
- **Pipeline Automatizado**: Orquestra o fluxo completo: buscar -> baixar -> treinar -> validar -> salvar -> fazer deploy -> testar.
- **Servidores Especializados**: Separa as responsabilidades de treinamento e deploy em servidores distintos.
- **Gerenciamento de Modelos**: Mantém um registro de modelos treinados e seus metadados.
- **Gerenciamento de Deployments**: Controla o deploy de modelos, criação de endpoints e monitoramento de status.
- **Exemplo de Ponta a Ponta**: Inclui um script de exemplo que demonstra todo o fluxo de trabalho.
- **Extensível**: A arquitetura modular permite adicionar novos `tools` e funcionalidades facilmente.

## Como Começar

Siga os passos abaixo para configurar e executar o projeto.

### Pré-requisitos

- Python 3.9 ou superior
- `pip` para gerenciamento de pacotes

### 1. Clonar o Repositório

```bash
gh repo clone <seu-usuario>/ml-mcp-agent
cd ml-mcp-agent
```

### 2. Instalar Dependências

É recomendado o uso de um ambiente virtual (`venv`) para isolar as dependências do projeto.

```bash
# Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar as dependências principais
pip install -r requirements.txt

# Instalar dependências dos servidores (incluindo kagglehub)
pip install -r servers/training_server/requirements.txt
pip install -r servers/deployment_server/requirements.txt
pip install -r servers/kaggle_server/requirements.txt
```

### 3. Configurar a API do Kaggle

Para que o servidor do Kaggle funcione, você precisa de um token de acesso. O método recomendado é usar um **token OAuth**, que é mais seguro.

**Método 1: Token OAuth (Recomendado)**

Você pode usar o token fornecido ou gerar um novo. Para configurar, crie um arquivo `.env` na raiz do projeto:

```bash
echo "KAGGLE_KEY=SEU_TOKEN_AQUI" > .env
```

O sistema carregará automaticamente esta variável de ambiente. O arquivo `.env` já está no `.gitignore` para evitar que seu token seja enviado para o repositório.

**Método 2: Arquivo `kaggle.json` (Legado)**

Se preferir o método tradicional com `kaggle.json`:

1.  Vá para `https://www.kaggle.com/account` e clique em **Create New API Token**.
2.  Mova o arquivo `kaggle.json` para `~/.kaggle/`.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Executar o Exemplo

#### Método Recomendado: Exemplo Standalone (Sem MCP)

Para garantir que **todas as métricas de performance sejam exibidas corretamente**, use o exemplo standalone:

```bash
python3 examples/standalone_training_example.py
```

Este exemplo:
- ✅ Baixa datasets do Kaggle automaticamente usando `kagglehub`
- ✅ Treina modelos com scikit-learn
- ✅ **Exibe todas as métricas de performance (accuracy, precision, recall, F1-score)**
- ✅ Mostra feature importance com gráficos
- ✅ Salva o modelo treinado
- ✅ Funciona sem dependência dos servidores MCP

**Saída esperada:**
```
📊 Test Metrics:
   val_accuracy: 1.0000 (100.00%)
   val_precision: 1.0000 (100.00%)
   val_recall: 1.0000 (100.00%)
   val_f1_score: 1.0000 (100.00%)

🔍 Feature Importance (sorted by importance):
   1. PetalLengthCm        0.4521 ██████████████████████
   2. PetalWidthCm         0.4234 █████████████████████
   3. SepalLengthCm        0.0823 ████
   4. SepalWidthCm         0.0422 ██
```

#### Método Alternativo: Exemplo com MCP

O projeto também inclui um exemplo completo que demonstra um pipeline real de ponta a ponta usando MCP:

1.  Iniciar o agente e os dois servidores MCP.
2.  Treinar um modelo de `RandomForest` com o dataset Iris.
3.  Validar o modelo.
4.  Fazer o deploy do modelo para um ambiente de "staging".
5.  Testar o endpoint de inferência.
6.  Listar modelos e deployments.
7.  Simular uma atualização e um rollback do modelo.
8.  Parar todos os processos.

Para executar o exemplo principal com treinamento real, use o script `real_training_example.py`:

```bash
# Criar um diretório de dados simulado para o exemplo
mkdir -p /tmp/data
# Nota: Os dados reais não estão incluídos, o código simula a leitura.

python3 examples/real_training_example.py
```

Você verá logs detalhados de cada etapa do processo, desde a inicialização dos servidores até a conclusão do pipeline.

## Como Funciona

O `agent/main.py` atua como o ponto de entrada principal. Ele inicializa o `MLPipelineOrchestrator`, que por sua vez instancia os `MCPClient`s para cada servidor.

O `MCPClient` (`agent/mcp_client.py`) é responsável por iniciar o processo do servidor Python correspondente e se comunicar com ele via `stdio`. Ele envia requisições JSON-RPC para invocar os `tools` definidos nos servidores e lê as respostas.

Os servidores (`servers/*/main.py`) são construídos com a biblioteca `FastMCP`. Cada função decorada com `@mcp.tool()` se torna uma capacidade que o agente pode invocar remotamente.

## Limitações e Próximos Passos

Este projeto agora implementa **treinamento real** de modelos de machine learning usando **scikit-learn**. As simulações foram substituídas por implementações funcionais.

Possíveis melhorias incluem:

- **Modelos Suportados**: Adicionar mais modelos de classificação e regressão ao `models.py`.
- **Deploy Real**: Utilizar `FastAPI` e `uvicorn` para expor endpoints de inferência reais e `Docker` para containerizar os modelos.
- **Transporte de Rede**: Mudar o transporte do MCP de `stdio` para `http` ou `websockets` para permitir a comunicação entre máquinas diferentes.
- **Segurança Aprimorada**: Implementar autenticação e autorização robustas entre o agente e os servidores.
- **Testes Unitários**: Adicionar testes para os `tools` dos servidores e para a lógica do orquestrador.

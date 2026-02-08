# Arquitetura do Sistema: Agente MCP para ML

## Visão Geral

Sistema de agente que orquestra treinamento e deploy de modelos de machine learning através de dois servidores MCP especializados.

## Componentes do Sistema

### 1. Servidor MCP de Datasets Kaggle (Kaggle Dataset Server)
- **Responsabilidade**: Buscar e baixar datasets do Kaggle
- **Tecnologia**: Python com FastMCP e Kaggle API
- **Tools Disponíveis**:
  - `search_datasets`: Busca datasets no Kaggle
  - `download_dataset`: Baixa um dataset do Kaggle
  - `list_dataset_files`: Lista os arquivos de um dataset
  - `get_dataset_metadata`: Obtém metadados de um dataset


### 2. Agente Principal (Host/Client)
- **Responsabilidade**: Orquestrar fluxo de treinamento e deploy
- **Tecnologia**: Python com SDK MCP
- **Funções**:
  - Receber requisições de usuário
  - Comunicar com servidor de treinamento
  - Comunicar com servidor de deploy
  - Gerenciar estado do pipeline

### 3. Servidor MCP de Treinamento (ML Training Server)
- **Responsabilidade**: Treinar modelos de machine learning
- **Tecnologia**: Python com FastMCP
- **Tools Disponíveis**:
  - `train_model`: Treina modelo com dados e hiperparâmetros
  - `validate_model`: Valida modelo em dataset de teste
  - `get_model_metrics`: Retorna métricas de performance
  - `list_trained_models`: Lista modelos treinados
  - `save_model`: Salva modelo em storage
  - `load_model`: Carrega modelo do storage

### 4. Servidor MCP de Deploy (ML Deployment Server)
- **Responsabilidade**: Fazer deploy de modelos em produção
- **Tecnologia**: Python com FastMCP
- **Tools Disponíveis**:
  - `deploy_model`: Deploy de modelo em ambiente
  - `create_endpoint`: Cria endpoint de inferência
  - `test_endpoint`: Testa endpoint com dados de exemplo
  - `get_deployment_status`: Status do deployment
  - `list_deployments`: Lista deployments ativos
  - `rollback_deployment`: Volta versão anterior
  - `update_model`: Atualiza modelo em produção

## Fluxo de Comunicação

```
┌─────────────────────────────────────────────────────────┐
│                    Agente Principal                      │
│              (ML Orchestration Agent)                    │
└────────────┬──────────────────────────────┬──────────────────────┬──────────────┘
             │                              │                      │
             │ MCP Protocol                 │ MCP Protocol         │ MCP Protocol
             │ (JSON-RPC)                   │ (JSON-RPC)           │ (JSON-RPC)
             │                              │                      │
    ┌────────▼──────────┐        ┌──────────▼────────┐      ┌───▼───────────────┐
    │  Training Server  │        │  Deploy Server    │      │  Kaggle Server      │
    │  (FastMCP)        │        │  (FastMCP)        │
    │                   │        │                   │
    │ - train_model     │        │ - deploy_model    │
    │ - validate_model  │        │ - create_endpoint │
    │ - get_metrics     │        │ - test_endpoint   │
    │ - get_metrics     │        │ - get_status      │      │ - search_datasets   │
    └───────────────────┘        └───────────────────┘      │ - download_dataset  │
                                                          └─────────────────────┘
```

## Fluxo de Trabalho Típico

1. **Busca e Download de Dataset (Novo)**
   - Agente busca por um dataset no Kaggle usando `search_datasets` no Kaggle Server
   - Agente baixa o dataset desejado com `download_dataset`, obtendo o caminho local dos arquivos

2. **Usuário solicita treinamento**
   - Agente recebe requisição, usando o caminho do dataset baixado
   - Agente chama `train_model` no Training Server

3. **Treinamento**
   - Training Server treina modelo
   - Retorna ID do modelo e métricas

4. **Validação**
   - Agente chama `validate_model` com dataset de teste
   - Avalia performance do modelo

5. **Deploy**
   - Se performance aceitável, agente chama `deploy_model` no Deploy Server
   - Deploy Server cria endpoint de inferência

6. **Teste**
   - Agente chama `test_endpoint` para validar deployment
   - Confirma que modelo está funcionando em produção

7. **Monitoramento**
   - Agente pode chamar `get_deployment_status` periodicamente
   - Monitora saúde do modelo em produção

## Estrutura de Diretórios

```
ml-mcp-agent/
├── README.md
├── requirements.txt
├── architecture.md
├── agent/
│   ├── main.py                 # Agente principal
│   ├── mcp_client.py           # Cliente MCP
│   └── orchestrator.py         # Orquestrador de pipeline
├── servers/
│   ├── kaggle_server/
│   │   ├── main.py             # Servidor de datasets Kaggle
│   │   └── requirements.txt
│   ├── training_server/
│   │   ├── main.py             # Servidor de treinamento
│   │   ├── models.py           # Lógica de treinamento
│   │   ├── storage.py          # Gerenciamento de storage
│   │   └── requirements.txt
│   └── deployment_server/
│       ├── main.py             # Servidor de deploy
│       ├── deployer.py         # Lógica de deployment
│       ├── endpoints.py        # Gerenciamento de endpoints
│       └── requirements.txt
├── examples/
│   ├── train_and_deploy.py     # Exemplo completo
│   └── config.yaml             # Configuração de exemplo
└── tests/
    ├── test_training_server.py
    └── test_deployment_server.py
```

## Tecnologias Utilizadas

- **Python 3.9+**: Linguagem principal
- **MCP SDK**: Protocolo de comunicação
- **FastMCP**: Framework para servidores MCP
- **Kaggle API**: Para interagir com a plataforma Kaggle
- **scikit-learn**: Machine learning
- **joblib**: Serialização de modelos
- **FastAPI**: Para endpoints de inferência
- **Docker**: Para containerização (opcional)

## Segurança

1. **Autenticação**: Tokens entre agente e servidores
2. **Autorização**: Validação de permissões para operações
3. **Validação de Entrada**: Sanitização de dados
4. **Logging**: Registro de todas as operações
5. **Isolamento**: Servidores em processos separados

## Escalabilidade

- Servidores podem rodar em máquinas diferentes
- Suporte para múltiplos modelos simultâneos
- Fila de treinamento para requisições
- Load balancing para endpoints de deploy

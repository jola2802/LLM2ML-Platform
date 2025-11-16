"""
Zentrale Netzwerkagenten-Konfiguration
"""

DEFAULT_MODEL = 'mistral:latest'

# Worker-Agents Konfiguration
WORKER_AGENTS = {
    'DATA_ANALYZER': {
        'key': 'DATA_ANALYZER',
        'name': 'Datenanalyse-Agent',
        'model': 'cogito:8b',
        'description': 'Analysiert Datasets und erstellt detaillierte Insights',
        'temperature': 0.2,
        'maxTokens': 4096,
        'retries': 3,
        'timeout': 60000,
        'category': 'analysis',
        'icon': '📊',
        'role': 'worker'
    },
    'HYPERPARAMETER_OPTIMIZER': {
        'key': 'HYPERPARAMETER_OPTIMIZER',
        'name': 'Hyperparameter Optimizer',
        'model': 'cogito:8b',
        'description': 'Suggests optimal hyperparameters and features',
        'temperature': 0.1,
        'maxTokens': 4096,
        'retries': 3,
        'timeout': 60000,
        'category': 'optimization',
        'icon': '⚙️',
        'role': 'worker'
    },
    'CODE_GENERATOR': {
        'key': 'CODE_GENERATOR',
        'name': 'Code Generator',
        'model': 'cogito:8b',
        'description': 'Generates Python code for ML training',
        'temperature': 0.1,
        'maxTokens': 8192,
        'retries': 3,
        'timeout': 120000,
        'category': 'generation',
        'icon': '💻',
        'role': 'worker'
    },
    'CODE_REVIEWER': {
        'key': 'CODE_REVIEWER',
        'name': 'Code Reviewer',
        'model': 'cogito:8b',
        'description': 'Reviews and optimizes generated code',
        'temperature': 0.1,
        'maxTokens': 8192,
        'retries': 3,
        'timeout': 120000,
        'category': 'review',
        'icon': '🔍',
        'role': 'worker'
    },
    'PERFORMANCE_ANALYZER': {
        'key': 'PERFORMANCE_ANALYZER',
        'name': 'Performance Analyzer',
        'model': 'cogito:8b',
        'description': 'Analyzes model performance',
        'temperature': 0.1,
        'maxTokens': 2048,
        'retries': 3,
        'timeout': 60000,
        'category': 'analysis',
        'icon': '📈',
        'role': 'worker'
    }
}

# Alle Agents
ALL_AGENTS = {**WORKER_AGENTS}

# Pipeline-Schritte
PIPELINE_STEPS = [
    {'step': 1, 'name': 'Datenanalyse', 'agent': 'DATA_ANALYZER', 'required': True},
    {'step': 2, 'name': 'Hyperparameter-Optimierung', 'agent': 'HYPERPARAMETER_OPTIMIZER', 'required': True},
    {'step': 3, 'name': 'Code-Generierung', 'agent': 'CODE_GENERATOR', 'required': True},
    # {'step': 4, 'name': 'Code-Review', 'agent': 'CODE_REVIEWER', 'required': False},
    # {'step': 5, 'name': 'Performance-Analyse', 'agent': 'PERFORMANCE_ANALYZER', 'required': False}
]

# Agent-Statistiken (vereinfacht)
_agent_stats = {
    'totalCalls': 0,
    'successfulCalls': 0,
    'failedCalls': 0
}

def get_agent_config(agent_key: str) -> dict:
    """Hole Agent-Konfiguration"""
    return ALL_AGENTS.get(agent_key, {})

def get_agent_model(agent_key: str) -> str:
    """Hole Agent-Modell"""
    config = get_agent_config(agent_key)
    return config.get('model', DEFAULT_MODEL)

def is_valid_agent(agent_key: str) -> bool:
    """Prüfe ob Agent existiert"""
    return agent_key in ALL_AGENTS

def log_agent_call(agent_key: str, model: str, purpose: str):
    """Logge Agent-Aufruf"""
    _agent_stats['totalCalls'] += 1
    print(f'🤖 {agent_key} ({model}): {purpose}')

def get_agent_stats() -> dict:
    """Hole Agent-Statistiken"""
    return _agent_stats.copy()


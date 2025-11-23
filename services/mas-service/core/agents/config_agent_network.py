"""
Zentrale Netzwerkagenten-Konfiguration mit Model-Tier-System
"""

import os

# Einheitliches Modell für alle Agents - vermeidet 200+ Sekunden Modell-Wechsel in Ollama
DEFAULT_MODEL = 'deepseek-r1:1.5b'

# Worker-Agents Konfiguration
WORKER_AGENTS = {
    'DATA_ANALYZER': {
        'key': 'DATA_ANALYZER',
        'name': 'Datenanalyse-Agent',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Analysiert Datasets und erstellt detaillierte Insights',
        'temperature': 0.2,
        'maxTokens': 4096,
        'retries': 3,
        'timeout': 90000, # 1,5min
        'category': 'analysis',
        'icon': '📊',
        'role': 'worker'
    },
    'DATA_CLEANER': {
        'key': 'DATA_CLEANER',
        'name': 'Data Cleaner',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Cleans and preprocesses data based on quality analysis',
        'temperature': 0.1,
        'maxTokens': 1024,
        'retries': 3,
        'timeout': 60000,
        'category': 'preprocessing',
        'icon': '🧹',
        'role': 'worker'
    },
    'FEATURE_ENGINEER': {
        'key': 'FEATURE_ENGINEER',
        'name': 'Feature Engineer',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Generiert neue Features aus vorhandenen Daten',
        'temperature': 0.2,
        'maxTokens': 4096,
        'retries': 3,
        'timeout': 90000, # 1,5min
        'category': 'engineering',
        'icon': '🔧',
        'role': 'worker'
    },
    'HYPERPARAMETER_OPTIMIZER': {
        'key': 'HYPERPARAMETER_OPTIMIZER',
        'name': 'Hyperparameter Optimizer',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Suggests optimal hyperparameters',
        'temperature': 0.1,
        'maxTokens': 2048,
        'retries': 3,
        'timeout': 60000,
        'category': 'optimization',
        'icon': '⚙️',
        'role': 'worker'
    },
    'CODE_GENERATOR': {
        'key': 'CODE_GENERATOR',
        'name': 'Code Generator',
        'model': 'deepseek-r1:1.5b',   # Kein LLM verwendet (Template-basiert)
        'description': 'Generates Python code from template (no LLM)',
        'temperature': 0.1,
        'maxTokens': 8192,
        'retries': 3,
        'timeout': 100000, # 100s
        'category': 'generation',
        'icon': '💻',
        'role': 'worker'
    },
    'CODE_REVIEWER': {
        'key': 'CODE_REVIEWER',
        'name': 'Code Reviewer',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Reviews generated code for correctness, security, and best practices',
        'temperature': 0.1,
        'maxTokens': 2048,
        'retries': 3,
        'timeout': 60000,
        'category': 'review',
        'icon': '👨‍💻',
        'role': 'worker'
    },
    'CODE_EXECUTOR': {
        'key': 'CODE_EXECUTOR',
        'name': 'Code Executor',
        'model': 'deepseek-r1:1.5b',
        'description': 'Executes Python code (no LLM)',
        'temperature': 0.0,
        'maxTokens': 0,
        'retries': 3,
        'timeout': 60000,
        'category': 'execution',
        'icon': '▶️',
        'role': 'worker'
    },
    'PERFORMANCE_ANALYZER': {
        'key': 'PERFORMANCE_ANALYZER',
        'name': 'Performance Analyzer',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Analyzes model performance',
        'temperature': 0.1,
        'maxTokens': 1024,
        'retries': 3,
        'timeout': 60000,
        'category': 'analysis',
        'icon': '📈',
        'role': 'worker'
    },
    'DECISION': {
        'key': 'DECISION',
        'name': 'Decision Agent',
        'model': 'deepseek-r1:1.5b',  # Einheitliches Modell für alle Agents
        'description': 'Decides if result is good enough or loop again',
        'temperature': 0.1,
        'maxTokens': 512,
        'retries': 3,
        'timeout': 60000,
        'category': 'decision',
        'icon': '🎯',
        'role': 'worker'
    }
}

# Alle Agents
ALL_AGENTS = {**WORKER_AGENTS}

# Pipeline-Schritte
PIPELINE_STEPS = [
    {'step': 1, 'name': 'Datenanalyse', 'agent': 'DATA_ANALYZER', 'required': True},
    {'step': 2, 'name': 'Feature Engineering', 'agent': 'FEATURE_ENGINEER', 'required': True},
    {'step': 3, 'name': 'Datenbereinigung', 'agent': 'DATA_CLEANER', 'required': False},
    {'step': 4, 'name': 'Hyperparameter-Optimierung', 'agent': 'HYPERPARAMETER_OPTIMIZER', 'required': True},
    {'step': 5, 'name': 'Code-Generierung', 'agent': 'CODE_GENERATOR', 'required': True},
    {'step': 6, 'name': 'Code-Review', 'agent': 'CODE_REVIEWER', 'required': False},
    {'step': 7, 'name': 'Code-Ausführung', 'agent': 'CODE_EXECUTOR', 'required': True},
    {'step': 8, 'name': 'Performance-Analyse', 'agent': 'PERFORMANCE_ANALYZER', 'required': True},
    {'step': 9, 'name': 'Entscheidung', 'agent': 'DECISION', 'required': True}
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

def get_agent_model(agent_key: str, use_tier: bool = None) -> str:
    """
    Hole Agent-Modell basierend auf Model-Tier-System
    
    Args:
        agent_key: Agent-Schlüssel
    
    Returns:
        Modell-Name für den Agent
    """
    config = get_agent_config(agent_key)
    
    # Fallback: Verwende konfiguriertes Modell
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

def get_model_tier_info(tier: str = None) -> dict:
    """
    Hole Informationen über Model-Tiers
    
    Args:
        tier: Optional - Spezifischer Tier (HIGH/MEDIUM/LOW)
    
    Returns:
        Tier-Konfiguration oder alle Tiers
    """
    if tier:
        return MODEL_TIERS.get(tier, {})
    return MODEL_TIERS.copy()

def get_agent_tier(agent_key: str) -> str:
    """Hole Model-Tier für einen Agent"""
    config = get_agent_config(agent_key)
    return config.get('modelTier', 'HIGH')

def get_agents_by_tier(tier: str) -> list:
    """Hole alle Agents mit einem bestimmten Model-Tier"""
    return [
        agent_key 
        for agent_key, config in ALL_AGENTS.items() 
        if config.get('modelTier') == tier
    ]

def get_optimization_summary() -> dict:
    """
    Erstelle Zusammenfassung der Model-Tier-Optimierung
    
    Returns:
        Dictionary mit Optimierungs-Statistiken
    """
    total_agents = len(ALL_AGENTS)
    llm_agents = [k for k, v in ALL_AGENTS.items() if v.get('modelTier') is not None]
    
    tier_counts = {
        'HIGH': len(get_agents_by_tier('HIGH')),
        'MEDIUM': len(get_agents_by_tier('MEDIUM')),
        'LOW': len(get_agents_by_tier('LOW'))
    }
    
    optimized_agents = tier_counts['MEDIUM'] + tier_counts['LOW']
    optimization_percentage = (optimized_agents / len(llm_agents) * 100) if llm_agents else 0
    
    return {
        'totalAgents': total_agents,
        'llmAgents': len(llm_agents),
        'tierCounts': tier_counts,
        'optimizedAgents': optimized_agents,
        'optimizationPercentage': round(optimization_percentage, 1),
        'useTiers': USE_MODEL_TIERS,
        'agentsByTier': {
            'HIGH': get_agents_by_tier('HIGH'),
            'MEDIUM': get_agents_by_tier('MEDIUM'),
            'LOW': get_agents_by_tier('LOW')
        }
    }


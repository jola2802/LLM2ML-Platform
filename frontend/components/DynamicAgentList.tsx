import React, { useState, useEffect } from 'react';

interface Agent {
  key: string;
  name: string;
  description: string;
  model: string;
  category: string;
  icon: string;
  temperature: number;
  maxTokens: number;
}

interface AgentCategory {
  name: string;
  agents: string[];
}

interface AgentConfig {
  agents: Agent[];
  categories: AgentCategory[];
  stats: {
    totalAgents: number;
    modelUsage: Record<string, number>;
    uniqueModels: number;
    defaultModel: string;
  };
  totalAgents: number;
}

interface DynamicAgentListProps {
  projectId: string;
  className?: string;
}

const DynamicAgentList: React.FC<DynamicAgentListProps> = ({
  projectId,
  className = ''
}) => {
  const [agentConfig, setAgentConfig] = useState<AgentConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Lade Agent-Konfiguration vom Backend
  const loadAgentConfig = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('/api/agents/frontend-config');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setAgentConfig(data.config);
        setError(null);
      } else {
        setError(data.error || 'Fehler beim Laden der Agent-Konfiguration');
      }
    } catch (err) {
      console.error('Fehler beim Laden der Agent-Konfiguration:', err);
      setError(err instanceof Error ? err.message : 'Unbekannter Fehler');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAgentConfig();
  }, []);

  // Kategorie-spezifische Farben
  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      analysis: 'bg-blue-50 border-blue-200 hover:bg-blue-100',
      code: 'bg-emerald-50 border-emerald-200 hover:bg-emerald-100',
      review: 'bg-purple-50 border-purple-200 hover:bg-purple-100',
      exploration: 'bg-amber-50 border-amber-200 hover:bg-amber-100',
      optimization: 'bg-red-50 border-red-200 hover:bg-red-100'
    };
    return colors[category] || 'bg-gray-50 border-gray-200 hover:bg-gray-100';
  };

  const getCategoryBadgeColor = (category: string) => {
    const colors: Record<string, string> = {
      analysis: 'bg-cyan-700/50 text-cyan-200 border border-cyan-500/50',
      code: 'bg-emerald-700/50 text-emerald-200 border border-emerald-500/50',
      review: 'bg-purple-700/50 text-purple-200 border border-purple-500/50',
      exploration: 'bg-amber-700/50 text-amber-200 border border-amber-500/50',
      optimization: 'bg-red-700/50 text-red-200 border border-red-500/50'
    };
    return colors[category] || 'bg-blue-700/50 text-blue-200 border border-blue-500/50';
  };

  // Loading-State
  if (isLoading) {
    return (
      <div className={`bg-blue-900/50 rounded-lg border border-blue-600/50 p-6 ${className}`}>
        <div className="flex items-center justify-center py-8">
          <div className="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          <span className="ml-3 text-blue-200 font-medium">Lade Agent-Konfiguration...</span>
        </div>
      </div>
    );
  }

  // Error-State
  if (error) {
    return (
      <div className={`bg-blue-900/50 rounded-lg border border-red-500/50 p-6 ${className}`}>
        <div className="text-center py-4">
          <div className="flex items-center justify-center w-12 h-12 bg-red-600/30 rounded-full mx-auto mb-3">
            <svg className="w-6 h-6 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-white font-medium mb-2">Konfiguration konnte nicht geladen werden</p>
          <p className="text-sm text-red-300 mb-4">{error}</p>
          <button
            onClick={loadAgentConfig}
            className="px-4 py-2 bg-cyan-600 text-white text-sm font-medium rounded-lg hover:bg-cyan-700 transition-colors"
          >
            Neu laden
          </button>
        </div>
      </div>
    );
  }

  // Keine Konfiguration verfügbar
  if (!agentConfig) {
    return (
      <div className={`bg-blue-900/50 rounded-lg border border-blue-600/50 p-6 ${className}`}>
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-blue-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-white mb-2">Keine Agent-Konfiguration</h3>
          <p className="text-sm text-blue-200">Die Agent-Konfiguration ist nicht verfügbar</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-blue-900/50 rounded-lg border border-blue-600/50 ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-blue-600/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-cyan-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Verfügbare Agents</h3>
              <p className="text-sm text-blue-200">
                {agentConfig.totalAgents} Agents • {agentConfig.stats.uniqueModels} verschiedene Modelle
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Agent-Kategorien */}
      <div className="p-6 space-y-6">
        {agentConfig.categories.map(category => {
          const categoryAgents = agentConfig.agents.filter(agent => 
            category.agents.includes(agent.key)
          );

          return (
            <div key={category.name}>
              <div className="flex items-center mb-3">
                <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${getCategoryBadgeColor(category.name).replace('bg-', 'bg-').replace('text-', 'text-')}`}>
                  {category.name.charAt(0).toUpperCase() + category.name.slice(1)}
                </span>
                <span className="ml-2 text-sm text-blue-300">
                  {categoryAgents.length} {categoryAgents.length === 1 ? 'Agent' : 'Agents'}
                </span>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {categoryAgents.map(agent => (
                  <div
                    key={agent.key}
                    className="p-4 rounded-lg border border-blue-600/50 bg-blue-800/50 hover:bg-blue-700/50 transition-all duration-200"
                  >
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0">
                        <span className="text-2xl">{agent.icon}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-semibold text-white truncate">
                          {agent.name}
                        </h4>
                        <p className="text-xs text-blue-200 mt-1 line-clamp-2">
                          {agent.description}
                        </p>
                        
                        {/* Modell-Informationen */}
                        <div className="mt-2 flex items-center justify-between">
                          <div className="text-xs">
                            <span className="inline-flex items-center px-2 py-1 bg-blue-700/70 text-blue-200 rounded-md font-mono">
                              {agent.model}
                            </span>
                          </div>
                          <div className="flex items-center space-x-3 text-xs text-blue-300">
                            <span title="Temperatur">T: {agent.temperature}</span>
                            <span title="Max Tokens">{Math.round(agent.maxTokens / 1000)}k</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer mit Statistiken */}
      <div className="px-6 py-4 bg-slate-800/30 border-t border-slate-600 rounded-b-lg">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-400">Standard-Modell:</span>
            <span className="ml-1 font-mono text-slate-200">{agentConfig.stats.defaultModel}</span>
          </div>
          <div className="text-right">
            <span className="text-slate-400">Modell-Verteilung: </span>
            <div className="inline-flex space-x-1 ml-1">
              {Object.entries(agentConfig.stats.modelUsage).map(([model, count]) => (
                <span key={model} className="inline-flex items-center px-2 py-1 bg-blue-600 text-blue-200 text-xs rounded-md">
                  {count}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DynamicAgentList;

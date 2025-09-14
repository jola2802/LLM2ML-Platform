import React, { useState, useEffect } from 'react';

interface AgentStatusIndicatorProps {
  projectId: string;
  className?: string;
  showDetails?: boolean;
}

interface ActiveAgent {
  agentKey: string;
  agentName: string;
  lastActivity?: {
    operation: string;
    startTime: string;
    status: string;
  };
}

const AgentStatusIndicator: React.FC<AgentStatusIndicatorProps> = ({
  projectId,
  className = '',
  showDetails = false
}) => {
  const [activeAgent, setActiveAgent] = useState<ActiveAgent | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch active agent status
  const fetchActiveAgent = async () => {
    try {
      const response = await fetch(`/api/agents/activities/${projectId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success && data.activeAgent) {
        setActiveAgent({
          agentKey: data.activeAgent,
          agentName: data.activeAgentName,
          lastActivity: data.activities?.[data.activities.length - 1]
        });
        setError(null);
      } else {
        setActiveAgent(null);
      }
    } catch (err) {
      console.error('Fehler beim Laden des Agent-Status:', err);
      setError(err instanceof Error ? err.message : 'Unbekannter Fehler');
      setActiveAgent(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh every 3 seconds
  useEffect(() => {
    fetchActiveAgent();
    const interval = setInterval(fetchActiveAgent, 3000);
    return () => clearInterval(interval);
  }, [projectId]);

  // Get agent icon based on agent type
  const getAgentIcon = (agentKey: string) => {
    switch (agentKey) {
      case 'DOMAIN_HYPERPARAMS':
        return '⚙️';
      case 'CODE_GENERATOR':
        return '🔧';
      case 'CODE_REVIEWER':
        return '🔍';
      case 'PERFORMANCE_ANALYST':
        return '📊';
      case 'DATA_EXPLORER':
        return '🔍';
      case 'AUTO_TUNER':
        return '🎯';
      default:
        return '🤖';
    }
  };

  // Format time for display
  const formatTimeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const seconds = Math.floor(diff / 1000);
    
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h`;
  };

  if (isLoading) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
        <span className="text-xs text-gray-400">Prüfe Agents...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
        <span className="text-xs text-red-400">Agent-Fehler</span>
      </div>
    );
  }

  if (!activeAgent) {
    return (
      <div className={`flex items-center space-x-3 ${className}`}>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-gray-500 rounded-full"></div>
          <span className="text-xl">😴</span>
        </div>
        <div>
          <span className="text-sm text-gray-400 font-medium">Keine aktiven Agents</span>
          <div className="text-xs text-gray-500">Alle Agents ruhen im Moment</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      <div className="flex items-center space-x-2">
        <div className="w-4 h-4 bg-green-500 rounded-full animate-pulse"></div>
        <span className="text-2xl">{getAgentIcon(activeAgent.agentKey)}</span>
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-green-300 truncate">
            {activeAgent.agentName}
          </span>
          <span className="text-xs px-2 py-1 bg-green-900/50 text-green-400 rounded">
            Aktiv
          </span>
          {activeAgent.lastActivity && (
            <span className="text-xs text-gray-400">
              vor {formatTimeAgo(activeAgent.lastActivity.startTime)}
            </span>
          )}
        </div>
        
        {showDetails && activeAgent.lastActivity && (
          <div className="text-xs text-gray-400 truncate mt-1">
            🔄 {activeAgent.lastActivity.operation}
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentStatusIndicator;

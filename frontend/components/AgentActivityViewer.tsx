import React, { useState, useEffect } from 'react';

interface AgentActivity {
  id: string;
  projectId: string;
  agentKey: string;
  agentName: string;
  operation: string;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  duration?: number;
  result?: any;
  nextAgent?: string;
  error?: string;
}

interface AgentActivityViewerProps {
  projectId: string;
  className?: string;
  showHeader?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const AgentActivityViewer: React.FC<AgentActivityViewerProps> = ({
  projectId,
  className = '',
  showHeader = true,
  autoRefresh = true,
  refreshInterval = 2000
}) => {
  const [activities, setActivities] = useState<AgentActivity[]>([]);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [activeAgentName, setActiveAgentName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>('');

  // Fetch agent activities
  const fetchActivities = async () => {
    try {
      const response = await fetch(`/api/agents/activities/${projectId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setActivities(data.activities || []);
        setActiveAgent(data.activeAgent);
        setActiveAgentName(data.activeAgentName);
        setLastUpdate(data.timestamp);
        setError(null);
      } else {
        setError(data.error || 'Fehler beim Laden der Agent-Activities');
      }
    } catch (err) {
      console.error('Fehler beim Laden der Agent-Activities:', err);
      setError(err instanceof Error ? err.message : 'Unbekannter Fehler');
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh effect
  useEffect(() => {
    fetchActivities();
    
    if (autoRefresh) {
      const interval = setInterval(fetchActivities, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [projectId, autoRefresh, refreshInterval]);

  // Status-Icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>;
      case 'completed':
        return <div className="w-3 h-3 bg-green-500 rounded-full"></div>;
      case 'failed':
        return <div className="w-3 h-3 bg-red-500 rounded-full"></div>;
      default:
        return <div className="w-3 h-3 bg-gray-500 rounded-full"></div>;
    }
  };

  // Status-Farbe
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'text-blue-400 bg-blue-900 border-blue-600';
      case 'completed':
        return 'text-green-400 bg-green-900 border-green-600';
      case 'failed':
        return 'text-red-400 bg-red-900 border-red-600';
      default:
        return 'text-gray-400 bg-gray-900 border-gray-600';
    }
  };

  // Formatiere Dauer
  const formatDuration = (duration?: number) => {
    if (!duration) return 'N/A';
    if (duration < 1000) return `${duration}ms`;
    return `${(duration / 1000).toFixed(1)}s`;
  };

  // Formatiere Zeit
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('de-DE', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  if (isLoading) {
    return (
      <div className={`p-4 bg-gray-800 rounded-lg border border-gray-700 ${className}`}>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-gray-300">Lade Agent-Activities...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`p-4 bg-red-900 rounded-lg border border-red-600 ${className}`}>
        <div className="flex items-center space-x-2">
          <span className="text-red-300">❌ Fehler:</span>
          <span className="text-red-200">{error}</span>
        </div>
        <button
          onClick={fetchActivities}
          className="mt-2 px-3 py-1 bg-red-700 text-red-200 rounded text-sm hover:bg-red-600"
        >
          Neu laden
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg border border-gray-700 ${className}`}>
      {showHeader && (
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <h3 className="text-lg font-semibold text-white">🤖 Agent-Activities</h3>
              {activeAgent && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <span className="text-sm text-blue-300">
                    Aktiv: {activeAgentName}
                  </span>
                </div>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">
                {activities.length} Activities
              </span>
              <button
                onClick={fetchActivities}
                className="p-1 text-gray-400 hover:text-white rounded"
                title="Aktualisieren"
              >
                🔄
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="p-4">
        {activities.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-2">🤖</div>
            <p>Noch keine Agent-Activities für dieses Projekt</p>
          </div>
        ) : (
          <div className="space-y-3">
            {activities.map((activity) => (
              <div
                key={activity.id}
                className={`p-3 rounded-lg border transition-all duration-200 ${getStatusColor(activity.status)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3 flex-1">
                    {getStatusIcon(activity.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-medium">{activity.agentName}</span>
                        <span className="text-xs px-2 py-1 rounded" style={{
                          backgroundColor: 'rgba(255,255,255,0.1)'
                        }}>
                          {activity.status}
                        </span>
                      </div>
                      <p className="text-sm opacity-80 mb-2">{activity.operation}</p>
                      
                      <div className="text-xs space-y-1 opacity-60">
                        <div>
                          🕐 Start: {formatTime(activity.startTime)}
                          {activity.endTime && (
                            <> → Ende: {formatTime(activity.endTime)}</>
                          )}
                        </div>
                        {activity.duration && (
                          <div>⏱️ Dauer: {formatDuration(activity.duration)}</div>
                        )}
                        {activity.nextAgent && (
                          <div>➡️ Weiter an: {activity.nextAgent}</div>
                        )}
                        {activity.error && (
                          <div className="text-red-300">❌ Fehler: {activity.error}</div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {lastUpdate && (
          <div className="mt-4 pt-3 border-t border-gray-700 text-xs text-gray-400 text-center">
            Letztes Update: {formatTime(lastUpdate)}
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentActivityViewer;

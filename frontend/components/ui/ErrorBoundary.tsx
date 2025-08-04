import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error details for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-[400px] flex items-center justify-center bg-slate-800/50 rounded-lg border border-red-500/50">
          <div className="text-center">
            <div className="text-red-400 text-6xl mb-4">⚠️</div>
            <h3 className="text-xl font-semibold text-white mb-2">Ein Fehler ist aufgetreten</h3>
            <p className="text-slate-400 text-sm mb-4">
              {this.state.error?.message || 'Ein unerwarteter Fehler ist aufgetreten.'}
            </p>
            
            {this.state.error?.stack && (
              <details className="text-left">
                <summary className="text-slate-300 cursor-pointer mb-2">Technische Details anzeigen</summary>
                <div className="bg-slate-900 rounded p-3 text-xs font-mono text-red-200 overflow-auto max-h-32">
                  {this.state.error.stack}
                </div>
              </details>
            )}
            
            <div className="mt-2 pt-2 border-t border-slate-700">
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors text-sm"
              >
                Seite neu laden
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
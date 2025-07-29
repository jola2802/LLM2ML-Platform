
import React, { useState, useEffect } from 'react';
import { Project, View, ProjectStatus, ModelType } from './types';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import ProjectWizard from './components/ProjectWizard';
import ProjectView from './components/ProjectView';
import { apiService, ApiProject } from './services/apiService';

// Hilfsfunktion um ApiProject zu Project zu konvertieren
const convertApiProjectToProject = (apiProject: ApiProject): Project => ({
  id: apiProject.id,
  name: apiProject.name,
  status: apiProject.status as ProjectStatus,
  modelType: apiProject.modelType as ModelType,
  dataSourceName: apiProject.dataSourceName,
  targetVariable: apiProject.targetVariable,
  features: apiProject.features,
  createdAt: apiProject.createdAt,
  performanceMetrics: apiProject.performanceMetrics,
  pythonCode: apiProject.pythonCode,
  originalPythonCode: apiProject.originalPythonCode,
  modelArtifact: apiProject.modelPath, // modelPath wird zu modelArtifact für Kompatibilität
  algorithm: apiProject.algorithm,
  hyperparameters: apiProject.hyperparameters,
  recommendations: apiProject.recommendations,
});

const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentView, setCurrentView] = useState<View>(View.DASHBOARD);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Projekte beim App-Start laden
  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setIsLoading(true);
      const apiProjects = await apiService.getProjects();
      const projects = apiProjects.map(convertApiProjectToProject);
      setProjects(projects);
    } catch (error) {
      console.error('Fehler beim Laden der Projekte:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Neues intelligentes Projekt erstellen (kommt vom neuen ProjectWizard)
  const handleCreateSmartProject = async (projectData: any) => {
    try {
      // Projekt wurde bereits im Backend erstellt, hier nur UI Updates
      const newProject = convertApiProjectToProject(projectData);
      
      setProjects(prev => [newProject, ...prev]);
      setCurrentView(View.DASHBOARD);
      
      // Polling für Updates starten
      pollForProjectUpdates(newProject.id);
    } catch (error) {
      console.error("Fehler beim Erstellen des intelligenten Projekts:", error);
    }
  };

  // Legacy: Altes manuelles Projekt erstellen (für Kompatibilität)
  const handleCreateProject = async (projectData: Omit<Project, 'id' | 'status' | 'createdAt'> & { csvFilePath?: string; algorithm?: string; hyperparameters?: object }) => {
    try {
      const createRequest = {
        name: projectData.name,
        modelType: projectData.modelType,
        dataSourceName: projectData.dataSourceName,
        targetVariable: projectData.targetVariable,
        features: projectData.features,
        csvFilePath: projectData.csvFilePath,
        algorithm: projectData.algorithm,
        hyperparameters: projectData.hyperparameters,
      };

      const apiProject = await apiService.createProject(createRequest);
      const newProject = convertApiProjectToProject(apiProject);
      
      setProjects(prev => [newProject, ...prev]);
      setCurrentView(View.DASHBOARD);
      
      // Polling für Updates starten
      pollForProjectUpdates(newProject.id);
    } catch (error) {
      console.error("Fehler beim Erstellen des Projekts:", error);
    }
  };

  // Projekt-Update Handler (für Code-Änderungen und Re-Training)
  const handleProjectUpdate = (updatedProject: Project) => {
    setProjects(prev => prev.map(p => 
      p.id === updatedProject.id ? updatedProject : p
    ));
    
    // Wenn Re-Training gestartet wurde, Polling starten
    if (updatedProject.status === ProjectStatus['Re-Training']) {
      pollForProjectUpdates(updatedProject.id);
    }
    
    // Selected project auch updaten falls es das gleiche ist
    if (selectedProject && selectedProject.id === updatedProject.id) {
      setSelectedProject(updatedProject);
    }
  };

  // Erweiterte Polling für Projekt-Updates (Training + Re-Training Status)
  const pollForProjectUpdates = async (projectId: string) => {
    const maxAttempts = 120; // 10 Minuten bei 5-Sekunden-Intervallen
    let attempts = 0;

    const poll = async () => {
      try {
        const apiProject = await apiService.getProject(projectId);
        const updatedProject = convertApiProjectToProject(apiProject);
        
        setProjects(prev => prev.map(p => 
          p.id === projectId ? updatedProject : p
        ));

        // Selected project auch updaten falls es das gleiche ist
        if (selectedProject && selectedProject.id === projectId) {
          setSelectedProject(updatedProject);
        }

        // Stop polling wenn Training/Re-Training abgeschlossen oder fehlgeschlagen
        if (updatedProject.status === ProjectStatus.Completed || 
            updatedProject.status === ProjectStatus.Failed ||
            updatedProject.status === ProjectStatus['Re-training Failed'] ||
            attempts >= maxAttempts) {
          console.log(`Polling für Projekt ${projectId} beendet. Status: ${updatedProject.status}`);
          return;
        }

        attempts++;
        setTimeout(poll, 5000); // 5 Sekunden warten
      } catch (error) {
        console.error('Fehler beim Polling:', error);
        // Bei Fehlern nach 3 Versuchen aufhören
        if (attempts >= 3) {
          return;
        }
        attempts++;
        setTimeout(poll, 10000); // Bei Fehlern länger warten
      }
    };

    setTimeout(poll, 3000); // Erstes Poll nach 3 Sekunden
  };

  const handleSelectProject = (project: Project) => {
    setSelectedProject(project);
    setCurrentView(View.PROJECT_VIEW);
  };
  
  const handleDeleteProject = async (projectId: string) => {
    try {
      await apiService.deleteProject(projectId);
      setProjects(prev => prev.filter(p => p.id !== projectId));
      
      // Falls das gelöschte Projekt gerade angezeigt wird, zurück zum Dashboard
      if (selectedProject && selectedProject.id === projectId) {
        handleBackToDashboard();
      }
    } catch (error) {
      console.error('Fehler beim Löschen des Projekts:', error);
    }
  };

  const handleBackToDashboard = () => {
    setSelectedProject(null);
    setCurrentView(View.DASHBOARD);
    // Projekte neu laden um aktuelle Status zu haben
    loadProjects();
  };

  const renderContent = () => {
    switch (currentView) {
      case View.WIZARD:
        return (
          <ProjectWizard 
            onBack={handleBackToDashboard} 
            onSubmit={handleCreateSmartProject} // Verwendet neuen Smart Project Handler
          />
        );
      case View.PROJECT_VIEW:
        return selectedProject ? (
          <ProjectView 
            project={selectedProject} 
            onBack={handleBackToDashboard}
            onProjectUpdate={handleProjectUpdate} // Callback für Project Updates
          />
        ) : (
          <Dashboard 
            projects={projects} 
            onCreate={() => setCurrentView(View.WIZARD)} 
            onSelectProject={handleSelectProject} 
            onDeleteProject={handleDeleteProject}
            isLoading={isLoading}
          />
        );
      case View.DASHBOARD:
      default:
        return (
          <Dashboard 
            projects={projects} 
            onCreate={() => setCurrentView(View.WIZARD)} 
            onSelectProject={handleSelectProject} 
            onDeleteProject={handleDeleteProject}
            isLoading={isLoading}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      <Header />
      <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderContent()}
      </main>

    </div>
  );
};

export default App;

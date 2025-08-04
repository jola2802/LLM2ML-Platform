
import React from 'react';
import { Project } from '../types';
import ProjectCard from './ProjectCard';
import { PlusIcon } from './icons/PlusIcon';

interface DashboardProps {
  projects: Project[];
  onCreate: () => void;
  onSelectProject: (project: Project) => void;
  onDeleteProject: (projectId: string) => void;
  isLoading: boolean;
}

const Dashboard: React.FC<DashboardProps> = ({ projects, onCreate, onSelectProject, onDeleteProject, isLoading }) => {
  if (isLoading) {
    return (
      <div className="animate-fade-in">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-3xl font-bold tracking-tight text-white">Projects Dashboard</h2>
        </div>
        <div className="text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
          <p className="mt-4 text-slate-400">Projekte werden geladen...</p>
        </div>
      </div>
    );
  }
  return (
    <div className="animate-fade-in">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold tracking-tight text-white">Projects Dashboard</h2>
        <button
          onClick={onCreate}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500 transition-colors"
        >
          <PlusIcon className="-ml-1 mr-2 h-5 w-5" />
          New Project
        </button>
      </div>

      {projects.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map(project => (
            <ProjectCard key={project.id} project={project} onSelect={onSelectProject} onDelete={onDeleteProject} />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 px-6 border-2 border-dashed border-slate-600 rounded-lg bg-slate-800/50">
          <svg className="mx-auto h-12 w-12 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
          </svg>
          <h3 className="mt-2 text-xl font-medium text-white">No projects yet</h3>
          <p className="mt-1 text-sm text-slate-400">Get started by creating a new machine learning project.</p>
          <div className="mt-6">
            <button
              onClick={onCreate}
              type="button"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500 transition-colors"
            >
              <PlusIcon className="-ml-1 mr-2 h-5 w-5" />
              New Project
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;


import React from 'react';
import { Project, ProjectStatus } from '../types';
import { PROJECT_STATUS_COLORS } from '../constants';
import { TrashIcon } from './icons/TrashIcon';

interface ProjectCardProps {
  project: Project;
  onSelect: (project: Project) => void;
  onDelete: (projectId: string) => void;
}

const ProjectCard: React.FC<ProjectCardProps> = ({ project, onSelect, onDelete }) => {
  const statusColor = PROJECT_STATUS_COLORS[project.status];

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to delete the project "${project.name}"?`)) {
        onDelete(project.id);
    }
  }

  return (
    <div 
      onClick={() => onSelect(project)}
      className="bg-slate-800 rounded-lg shadow-lg overflow-hidden cursor-pointer group transform hover:-translate-y-1 transition-all duration-300 flex flex-col justify-between border border-slate-700 hover:border-slate-600"
    >
      <div className="p-5">
        <div className="flex justify-between items-start">
            <span className={`px-3 py-1 text-xs font-semibold rounded-full ${statusColor}`}>
              {project.status}
            </span>
            {project.status === ProjectStatus.Training && (
                <div className="w-4 h-4">
                    <span className="relative flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
                    </span>
                </div>
            )}
        </div>
        <h3 className="text-xl font-bold text-white mt-3 truncate">{project.name}</h3>
        <p className="text-sm text-slate-400 mt-1">Model: {project.modelType}</p>
        <p className="text-sm text-slate-400">Data: {project.dataSourceName}</p>
      </div>
      <div className="bg-slate-800/50 px-5 py-3 flex justify-between items-center border-t border-slate-700">
        <p className="text-xs text-slate-500">
          Created: {new Date(project.createdAt).toLocaleDateString()}
        </p>
        <button 
            onClick={handleDelete}
            className="p-1 text-slate-500 hover:text-red-500 transition-colors rounded-full opacity-0 group-hover:opacity-100"
            aria-label="Delete project"
            >
            <TrashIcon className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default ProjectCard;

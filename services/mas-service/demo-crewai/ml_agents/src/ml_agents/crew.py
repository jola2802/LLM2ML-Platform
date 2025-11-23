from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ml_agents.src.ml_agents.tools import CsvReaderTool, train_model_tool
import os
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Ollama LLM Konfiguration
OLLAMA_BASE_URL = os.getenv('API_BASE', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('MODEL', 'llama3.2:latest')

# Erstelle Ollama LLM Instanz
# Format: "ollama/<modellname>" für CrewAI
llm=LLM(
    model="ollama/llama3.2", 
    base_url="http://localhost:11434",
)


@CrewBase
class MlAgents():
    """MlAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'], # type: ignore[index]
            tools=[CsvReaderTool()],
            llm=llm,
            verbose=True
        )

    @agent
    def data_cleaner(self) -> Agent:
        return Agent(
            config=self.agents_config['data_cleaner'], # type: ignore[index]
            tools=[CsvReaderTool()],
            llm=llm,
            verbose=True
        )
        
    @agent
    def feature_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config['feature_engineer'], # type: ignore[index]
            tools=[CsvReaderTool()],
            llm=llm,
            verbose=True
        )

    @agent
    def algorithm_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['algorithm_selector'], # type: ignore[index]
            llm=llm,
            verbose=True
        )
        
    @agent
    def hyperparameter_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config['hyperparameter_optimizer'], # type: ignore[index]
            llm=llm,
            verbose=True
        )
        
    @agent
    def performance_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['performance_analyzer'], # type: ignore[index]
            tools=[train_model_tool()],
            llm=llm,
            verbose=True
        )
        
    @agent
    def decision_maker(self) -> Agent:
        return Agent(
            config=self.agents_config['decision_maker'], # type: ignore[index]
            llm=llm,
            verbose=True
        )
        
    @task
    def data_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_analysis_task'], # type: ignore[index]
        )

    @task
    def data_cleaning_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_cleaning_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def feature_engineering_task(self) -> Task:
        return Task(
            config=self.tasks_config['feature_engineering_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def algorithm_selection_task(self) -> Task:
        return Task(
            config=self.tasks_config['algorithm_selection_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def hyperparameter_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config['hyperparameter_optimization_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def performance_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['performance_analysis_task'], # type: ignore[index]
            output_file='report.md'
        )

    @task
    def decision_task(self) -> Task:
        return Task(
            config=self.tasks_config['decision_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

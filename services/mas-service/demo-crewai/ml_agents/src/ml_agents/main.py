#!/usr/bin/env python
import sys
import warnings
import os
from pathlib import Path

from datetime import datetime

from ml_agents.crew import MlAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew.
    """
    # Pfad zur CSV-Datei relativ zum Hauptverzeichnis
    # Von demo-crewai/ml_agents/ aus geht es zwei Ebenen hoch und dann zu test_dataset.csv
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    csv_path = base_dir / "test_dataset.csv"
    
    # Falls die Datei nicht gefunden wird, versuche absoluten Pfad
    if not csv_path.exists():
        # Versuche es im aktuellen Verzeichnis
        csv_path = Path("test_dataset.csv")
        if not csv_path.exists():
            # Versuche es relativ zum mas-service Verzeichnis
            csv_path = Path("../../test_dataset.csv")
    
    inputs = {
        'topic': 'Machine Learning Modell für Kreditwürdigkeit',
        'current_year': str(datetime.now().year),
        'csv_path': str(csv_path.absolute())
    }

    try:
        MlAgents().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    csv_path = base_dir / "test_dataset.csv"
    if not csv_path.exists():
        csv_path = Path("test_dataset.csv")
        if not csv_path.exists():
            csv_path = Path("../../test_dataset.csv")
    
    inputs = {
        "topic": "Machine Learning Modell für Kreditwürdigkeit",
        'current_year': str(datetime.now().year),
        'csv_path': str(csv_path.absolute())
    }
    try:
        MlAgents().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MlAgents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    base_dir = Path(__file__).parent.parent.parent.parent.parent
    csv_path = base_dir / "test_dataset.csv"
    if not csv_path.exists():
        csv_path = Path("test_dataset.csv")
        if not csv_path.exists():
            csv_path = Path("../../test_dataset.csv")
    
    inputs = {
        "topic": "Machine Learning Modell für Kreditwürdigkeit",
        "current_year": str(datetime.now().year),
        "csv_path": str(csv_path.absolute())
    }

    try:
        MlAgents().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = MlAgents().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")

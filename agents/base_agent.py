from __future__ import annotations
import json
from typing import Callable, Dict, Any, Optional
from langchain_community.llms import Ollama
from utils.logging_config import AppLogger

class AgentResult:
    """A standardized class for returning results from an agent."""
    def __init__(self, success: bool, text: str, metadata: Optional[Dict[str, Any]] = None, plot_data: Optional[Dict[str, Any]] = None, plot_type: Optional[str] = None):
        self.success = success
        self.text = text
        self.metadata = metadata or {}
        self.plot_data = plot_data
        self.plot_type = plot_type

    def to_json(self) -> str:
        return json.dumps({"success": self.success, "text": self.text, "metadata": self.metadata, "plot_data": self.plot_data, "plot_type": self.plot_type}, indent=2)

class BaseAgent:
    """Abstract base class for all agents."""
    def __init__(self, name: str, llm: Ollama, tools: Dict[str, Callable], logger: AppLogger):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.logger = logger

    def handle(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        raise NotImplementedError("Each agent must implement its own handle method.")


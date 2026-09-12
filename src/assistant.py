import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from schemas import SessionState
from retrieval import SimulatedRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow, AgentState


class DocumentAssistant:
    """
    The assistant creates and loads sessions and
    stores state/session data within a file.
    """

    def __init__(
            self,
            openai_api_key: str,
            model_name: str = "gpt-4o-mini",
            temperature: float = 0.1,
            session_storage_path: str = "./sessions",
            base_url: Optional[str] = None,
    ):
        llm_kwargs = {
            "api_key": openai_api_key,
            "model": model_name,
            "temperature": temperature,
        }
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if resolved_base_url:
            llm_kwargs["base_url"] = resolved_base_url

        self.llm = ChatOpenAI(**llm_kwargs)

        self.retriever = SimulatedRetriever()
        self.logs_dir = "./logs"
        self.tool_logger = ToolLogger(logs_dir=self.logs_dir)
        self.tools = get_all_tools(self.retriever, self.tool_logger)

        self.workflow = create_workflow(self.llm, self.tools)

        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.current_session: Optional[SessionState] = None

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start a new session or resume an existing one."""
        if session_id and self._session_exists(session_id):
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[]
            )
            print(f"Started new session {session_id}")
        self._bind_session_logger(self.current_session.session_id)
        self._save_session()
        return self.current_session.session_id

    def _bind_session_logger(self, session_id: str) -> None:
        """Rebind ToolLogger and tools so each session writes its own log file."""
        self.tool_logger = ToolLogger(logs_dir=self.logs_dir, session_id=session_id)
        self.tools = get_all_tools(self.retriever, self.tool_logger)

    def _session_exists(self, session_id: str) -> bool:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        return os.path.exists(filepath)

    def _load_session(self, session_id: str) -> SessionState:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return SessionState(**data)

    @staticmethod
    def _iso(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _plain_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        return json.dumps(content, default=str)

    def _json_safe_history(self, history: List[Any]) -> List[Dict[str, Any]]:
        safe: List[Dict[str, Any]] = []
        for turn in history or []:
            if not isinstance(turn, dict):
                continue
            safe.append({
                "user_input": turn.get("user_input"),
                "intent": turn.get("intent"),
                "tools_used": list(turn.get("tools_used") or []),
                "summary": turn.get("summary") or "",
                "messages": list(turn.get("messages") or []),
            })
        return safe

    def _save_session(self) -> None:
        if not self.current_session:
            return
        filepath = os.path.join(
            self.session_storage_path,
            f"{self.current_session.session_id}.json"
        )
        payload = {
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "document_context": list(self.current_session.document_context or []),
            "created_at": self._iso(self.current_session.created_at),
            "last_updated": self._iso(self.current_session.last_updated),
            "conversation_history": self._json_safe_history(
                self.current_session.conversation_history
            ),
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    def _get_conversation_summary(self, config) -> str:
        try:
            current_state = self.workflow.get_state(config).values or {}
            return current_state.get("conversation_summary") or "No previous conversation."
        except Exception:
            return "No previous conversation."

    def _get_conversation_history(self, config) -> List[BaseMessage]:
        try:
            current_state = self.workflow.get_state(config).values or {}
            return current_state.get("messages", []) or []
        except Exception:
            return []

    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process a user message using the LangGraph workflow."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        config = {
            "configurable": {
                "thread_id": self.current_session.session_id,
                "llm": self.llm,
                "tools": self.tools,
            }
        }

        initial_state: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": self._get_conversation_summary(config),
            "active_documents": self.current_session.document_context,
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            "actions_taken": [],
        }
        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            intent = final_state.get("intent")
            intent_data = intent.model_dump() if intent else None
            messages = final_state.get("messages") or []
            messages_safe = [
                {
                    "role": getattr(message, "type", message.__class__.__name__),
                    "content": self._plain_text(message),
                }
                for message in messages
            ]
            summary = final_state.get("conversation_summary") or ""
            tools_used = list(final_state.get("tools_used") or [])

            self.current_session.conversation_history.append({
                "user_input": user_input,
                "intent": intent_data,
                "tools_used": tools_used,
                "summary": summary,
                "messages": messages_safe,
            })
            self.current_session.last_updated = datetime.now()
            if final_state.get("active_documents"):
                self.current_session.document_context = list(set(
                    self.current_session.document_context +
                    final_state["active_documents"]
                ))
            self._save_session()

            last_content = messages_safe[-1]["content"] if messages_safe else None
            return {
                "success": True,
                "response": last_content,
                "intent": intent_data,
                "tools_used": tools_used,
                "sources": final_state.get("active_documents", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": summary,
            }
        except Exception as e:
            self.current_session.conversation_history.append({
                "user_input": user_input,
                "intent": None,
                "tools_used": [],
                "summary": "",
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"Error: {e}"},
                ],
            })
            self.current_session.last_updated = datetime.now()
            self._save_session()
            return {
                "success": False,
                "error": str(e),
                "response": None
            }

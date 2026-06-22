from pathlib import Path
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

PLANS_DIR = Path(".fraude/plans")

class Plan(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    status: str
    content: str

    @property
    def path(self) -> str:
        return (PLANS_DIR / f"{self.id}.json").as_posix()

    def to_markdown(self) -> str:
        return self.content


def save_plan(plan: Plan, plans_dir: Path) -> Path:
    plans_dir.mkdir(parents=True, exist_ok=True)
    path = plans_dir / f"{plan.id}.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(plan.model_dump_json(indent=2))
    return path


def load_plan(plan_id: str, plans_dir: Path) -> Plan:
    normalized_id = UUID(plan_id)
    path = plans_dir / f"{normalized_id}.json"
    return Plan.model_validate_json(path.read_text(encoding="utf-8"))

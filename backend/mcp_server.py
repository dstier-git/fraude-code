import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
import planning

WORKSPACE = Path(os.environ.get("FRAUDE_WORKSPACE", Path.cwd())).resolve()
mcp = FastMCP("Fraude Code", json_response=True)


def safe_path(relative_path: str) -> Path:
    candidate = (WORKSPACE / relative_path).resolve()

    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError("Path is outside the workspace")
    return candidate


def plans_dir() -> Path:
    return safe_path(planning.PLANS_DIR.as_posix())


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the configured workspace."""
    return safe_path(path).read_text(encoding="utf-8")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Create a new UTF-8 text file; never overwrite an existing file."""
    file = safe_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return f"Successfully wrote {len(content.splitlines())} lines to {path}."

@mcp.tool()
def create_plan(title: str, content: str) -> str:
    """Create and persist a draft plan inside the configured workspace."""
    plan = planning.Plan(status="draft", title=title, content=content)
    planning.save_plan(plan, plans_dir())
    return f"Successfully created plan {title} at {plan.path}."


@mcp.tool()
def get_plan(plan_id: str) -> str:
    """Return a persisted plan's content by its UUID."""
    return planning.load_plan(plan_id, plans_dir()).to_markdown()


if __name__ == "__main__":
    mcp.run(transport="stdio")

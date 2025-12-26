# BenchAI Tools Reference

Complete list of all 88+ tools available in BenchAI.

## Tool Categories

| Category | Count | Description |
|----------|-------|-------------|
| Infrastructure | 9 | Docker, system monitoring |
| File Operations | 8 | Read, write, edit files |
| Git | 8 | Git operations |
| GitHub CLI | 6 | PRs, issues, repos |
| Learning | 6 | Flashcards, quizzes |
| Code | 6 | Code execution, review |
| Memory | 6 | Remember, recall |
| Web | 4 | Search, fetch |
| Obsidian | 6 | Notes integration |
| Data | 4 | Excel, CSV, analysis |
| API | 2 | HTTP requests |
| Diagrams | 2 | Mermaid diagrams |
| Misc | 10+ | TTS, vision, etc |

---

## Infrastructure Tools

### docker_ps
List running Docker containers.

**Trigger:** "containers", "docker", "what's running"

**Example:** "what containers are running?"

### docker_logs
Get logs from a container.

**Trigger:** "logs", "container logs"

**Example:** "get nginx logs"

### docker_control
Start, stop, restart containers.

**Trigger:** "restart", "stop", "start"

**Example:** "restart postgres"

### docker_stats
Get container resource usage.

**Trigger:** "docker stats", "container stats"

### homelab_status
Full homelab health report.

**Trigger:** "homelab status", "system status"

### service_health
Check health of all services.

**Trigger:** "service health", "are services healthy"

### system_info
Get system information (CPU, RAM, disk).

**Trigger:** "system info", "specs"

### gpu_status
Check GPU usage and memory.

**Trigger:** "gpu", "nvidia", "vram"

---

## File Operations

### read_file
Read file contents.

**Example:** "read ~/project/config.py"

### write_file
Write/create a file.

**Example:** "create a file ~/test.py with hello world"

### edit_file
Edit a file with find/replace.

**Example:** "in config.py, change debug=False to debug=True"

### list_files
List directory contents.

**Example:** "list files in ~/project"

### search_files
Search for files by pattern.

**Example:** "find all .py files in ~/project"

### file_info
Get file metadata.

### delete_file
Delete a file (with confirmation).

### move_file
Move or rename a file.

---

## Git Tools

### git_status
Get repository status.

**Example:** "git status in ~/myproject"

### git_diff
Show uncommitted changes.

**Example:** "git diff"

### git_log
Show commit history.

**Example:** "show last 10 commits"

### git_commit
Create a commit.

**Example:** "commit with message: Fix bug"

### git_push
Push to remote.

**Example:** "push to origin"

### git_pull
Pull from remote.

### git_create_branch
Create a new branch.

**Example:** "create branch feature-x"

### git_checkout
Switch branches.

---

## GitHub CLI Tools

### gh_pr_create
Create a pull request.

**Example:** "create a PR with title: Add feature"

### gh_pr_list
List pull requests.

**Example:** "list open PRs"

### gh_pr_merge
Merge a pull request.

**Example:** "merge PR 123"

### gh_issue_create
Create an issue.

**Example:** "create issue: Bug in login"

### gh_issue_list
List issues.

### gh_repo_info
Get repository information.

---

## Learning Tools

### create_flashcard
Create a study flashcard.

**Example:** "create flashcard: Q: What is a closure? A: A function that..."

### study_flashcards
Review flashcards by topic.

**Example:** "study my JavaScript flashcards"

### generate_quiz
Generate a quiz on a topic.

**Example:** "quiz me on Python with 5 questions"

### learning_path
Generate a learning roadmap.

**Example:** "learning path for machine learning"

### code_challenge
Get a coding challenge.

**Example:** "give me a recursion challenge"

### explain_concept
Explain a programming concept.

**Example:** "explain async/await for beginners"

---

## Code Tools

### code_task
Write or modify code.

**Example:** "write a function to sort a list"

### run_code
Execute Python code.

**Example:** "run: print(2+2)"

### code_review
Review code for issues.

**Example:** "review this function: [code]"

### generate_tests
Generate unit tests.

**Example:** "write tests for this function"

### explain_error
Explain an error message.

**Example:** "explain: TypeError: 'NoneType'..."

### refactor_code
Suggest refactoring.

---

## Memory Tools

### remember
Save information to memory.

**Example:** "remember that I prefer tabs"

### recall
Search memory for information.

**Example:** "what do you remember about my preferences?"

### forget
Delete a memory.

### list_memories
List all memories.

### search_knowledge
Semantic search of memory.

### memory_stats
Get memory statistics.

---

## Web Tools

### web_search
Search the web via SearXNG.

**Example:** "search the web for Python tutorials"

### fetch_url
Fetch and summarize a URL.

**Example:** "summarize https://example.com"

### extract_data
Extract data from a webpage.

### api_request
Make an HTTP request.

---

## Obsidian Tools

### obsidian_read
Read a note from Obsidian.

**Example:** "read my daily note"

### obsidian_write
Write/update a note.

**Example:** "save this to obsidian: Meeting notes..."

### obsidian_search
Search notes.

**Example:** "search my notes for project ideas"

### obsidian_list
List notes in a folder.

### obsidian_create
Create a new note.

### obsidian_append
Append to a note.

---

## Data Tools

### read_csv
Read and analyze CSV file.

### read_excel
Read Excel file.

### analyze_data
Analyze data with pandas.

### create_excel
Create an Excel file.

---

## API Tools

### api_request
Make HTTP request.

**Example:** "GET https://api.example.com/data"

### api_test_suite
Run API tests.

---

## Diagram Tools

### create_diagram
Create a Mermaid diagram.

**Example:** "create a flowchart for login process"

### flowchart
Create a flowchart.

---

## Other Tools

### speak
Text-to-speech.

**Example:** "say hello world"

### vision_analysis
Analyze an image.

**Example:** "what's in this image: [path]"

### generate_image
Generate an image (requires ComfyUI).

**Example:** "generate an image of a sunset"

### reasoning
Deep reasoning for complex problems.

### shell_command
Execute safe shell commands.

### calculator
Perform calculations.

---

## Tool Selection

BenchAI automatically selects the best tool based on your query. Keywords help:

- **Docker:** "containers", "docker", "restart"
- **Git:** "git status", "commit", "push"
- **Web:** "search the web", "latest", "current"
- **Memory:** "remember", "recall", "what did I say"
- **Files:** "read", "write", "edit", "create file"
- **Code:** "write code", "run", "review"
- **Learning:** "flashcard", "quiz", "explain"

---

## Adding Custom Tools

To add a new tool, edit `llm_router.py`:

```python
TOOLS["my_tool"] = {
    "name": "my_tool",
    "description": "What this tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."}
        },
        "required": ["param1"]
    }
}

async def execute_my_tool(params):
    # Implementation
    return {"result": "..."}
```

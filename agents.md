# AGENTS.md

When making code changes:
- Group edits into small, logically coherent commits.
- Do not mix refactors, bug fixes, and formatting in the same commit.
- After each logical unit is complete and validated, stage only the relevant files.
- Create a git commit with a concise conventional-commit style message.
- If the requested task spans multiple concerns, split it into multiple commits and explain the proposed commit boundaries before committing.

# Commit policy
- Prefer multiple atomic commits over one large commit.
- Each commit must correspond to one described change.
- Separate behavior changes, refactors, dependency updates, and tests.
- Before each commit, show the files included and a one-line rationale.
- Use Conventional Commits:
  - feat:
  - fix:
  - refactor:
  - test:
  - docs:
- Never commit unrelated formatting changes with functional edits.
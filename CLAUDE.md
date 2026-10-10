@AGENTS.md

Claude-specific: `/check` (`.claude/skills/check/`) runs the checks above; there are no
path-scoped rules or hooks, the repo is too small to need them. Your own notes go in
`CLAUDE.local.md` (ignored).
Multi-step work: the user-level `/autonomous-delivery` skill and its `delivery-reviewer` agent
follow this file's rules.

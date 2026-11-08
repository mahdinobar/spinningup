# Repository instructions

## Disposable files and 5 GB working budget

- Store all generated disposable files and controllable caches under
  `/home/mahdi/codes/spinningup/.codex-tmp/<task-id>/`. Create a task directory
  only when needed. Set `TMPDIR`, `TMP`, and `TEMP` to it for commands that
  generate temporary files; configure other cache and output paths there too.
- Keep the total `.codex-tmp` directory below **5,000,000,000 bytes (5 GB)**.
  This repository-specific budget supersedes the general 10 GB warning
  threshold. It is a Codex working rule, not a filesystem quota.
- Measure the total size before substantial output, periodically during long
  tasks, and before finishing. Check that planned output fits the remaining
  budget before starting it. Reserve space for tools' intermediate files.
- At 4,000,000,000 bytes, clean up the current task's unneeded intermediates
  before generating more bulk output. If an operation could exceed 5 GB,
  reduce or split its output, clean verified disposable files, or stop that
  operation and explain the limitation. Never delete user data to make room.
- Before completing a task, verify exact paths and remove its disposable
  files and empty task directory. Keep requested deliverables outside
  `.codex-tmp`; identify any intentionally retained temporary files and why.
- Older task directories may be removed only after verifying that the task
  has ended, their contents are disposable, and any requested deliverables
  are preserved outside `.codex-tmp`. Preserve active tasks, user data,
  original results, and anything whose retention is uncertain. Do not use
  blanket deletion, modification time alone, or follow symlinks to choose
  files for cleanup.
- Perform these checks during active work; do not install background cleanup
  jobs or change filesystem quotas without an explicit request.

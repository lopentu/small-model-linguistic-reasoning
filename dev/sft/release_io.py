"""Checks for explicit uploads of restricted training inputs or weights."""
def require_private_repo(api, repo_id, *, repo_type):
    if not repo_id:
        raise SystemExit("Set an explicit upload destination")
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise SystemExit("Upload destination must be private")

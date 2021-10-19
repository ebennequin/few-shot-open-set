from st_constants import GIT_REPO


def get_commit_message(rev: str) -> str:
    return GIT_REPO.commit(rev).summary

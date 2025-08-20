def format_prompt(observation: str) -> str:
    """
    Formats the observation string into a prompt suitable for the policy.
    """
    return f"{observation.strip()}\n>"
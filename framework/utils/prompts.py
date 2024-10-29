# utils/prompts.py
from typing import List, Optional, NamedTuple
import os


class Prompt(NamedTuple):
    category: str
    text: str


class PromptCollection:
    def __init__(self):
        self.prompts: List[Prompt] = []
        self.categories: List[str] = []

    def add_prompt(self, category: str, text: str):
        self.prompts.append(Prompt(category, text))
        if category not in self.categories:
            self.categories.append(category)

    def get_prompts(self, categories: Optional[List[str]] = None) -> List[Prompt]:
        """Get prompts, optionally filtered by categories."""
        if categories is None:
            return self.prompts
        return [p for p in self.prompts if p.category in categories]

    def get_categories(self) -> List[str]:
        return self.categories


def load_prompts(file_path: str) -> PromptCollection:
    """
    Load categorized prompts from a text file.

    Format:
    Category1
    prompt1
    prompt2

    Category2
    prompt3
    prompt4

    Categories are single lines that follow blank lines (or start of file).
    All other non-blank lines are prompts belonging to the current category.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    prompts = PromptCollection()
    current_category = None
    previous_line_blank = True  # Consider start of file as after a blank line

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if not line:  # Blank line
                previous_line_blank = True
                continue

            if previous_line_blank:  # Line after blank is a category
                current_category = line
                previous_line_blank = False
            else:  # All other non-blank lines are prompts
                if current_category:  # Should always be true after first category
                    prompts.add_prompt(current_category, line)
                previous_line_blank = False

    return prompts
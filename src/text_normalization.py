import re
import unicodedata


MULTISPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


def strip_accents(text: str) -> str:
    """
    Remove accent marks / combining marks from text.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_mapper_text(text: str) -> str:
    """
    Normalize product/category text for category matching.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = strip_accents(text)
    text = NON_ALNUM_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()

    return text


def normalize_aliases(aliases: str) -> str:
    """
    Normalize a comma-separated alias field into a space-separated text block.
    """
    if aliases is None:
        return ""

    parts = [normalize_mapper_text(part) for part in str(aliases).split(",")]
    parts = [part for part in parts if part]
    return " ".join(parts)


def build_category_match_text(
    category_name: str,
    category_path: str = "",
    aliases: str = "",
) -> str:
    """
    Build a searchable category text field from name, path, and aliases.
    """
    normalized_name = normalize_mapper_text(category_name)
    normalized_path = normalize_mapper_text(category_path)
    normalized_aliases = normalize_aliases(aliases)

    return " ".join(
        part for part in [normalized_name, normalized_path, normalized_aliases] if part
    )
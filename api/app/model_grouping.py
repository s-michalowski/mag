import re


KNOWN_GROUP_PREFIXES = {
    "CARBIDES": "CARBIDES",
    "CRACKS": "CRACKS",
    "DARK_VAC": "DARK_VAC",
    "DECARBURIZATION": "DECARBURIZATION",
    "DEND": "DENDRITES",
    "DENDRITES": "DENDRITES",
    "FERRITE": "FERRITE",
    "GRAIN": "GRAIN",
    "GROUND": "GROUND",
    "LAYER": "LAYER",
    "MARTENSITE": "MARTENSITE",
    "OXI": "OXIDATION",
    "OXIDATION": "OXIDATION",
    "PEARLITE": "PEARLITE",
    "RETAINED": "RETAINED_AUSTENITE",
    "SEGMENT": "SEGMENT",
}


def normalize_text(name: str) -> str:
    name = str(name).strip().upper()
    name = name.replace(" ", "")
    name = name.replace("-", "_")
    return name


def extract_auto_group(name: str) -> str:
    """
    Dla nowych nazw, których nie ma w znanych grupach:
    - bierzemy początek nazwy do pierwszego "_" lub do pierwszej cyfry
    - jeśli nic sensownego nie wyjdzie, zwracamy całą nazwę
    """
    normalized = normalize_text(name)

    match = re.match(r"^[A-Z]+", normalized)
    if match:
        return match.group(0)

    return normalized if normalized else "UNKNOWN"


def get_model_group(model_name: str) -> str:
    if model_name is None:
        return "UNKNOWN"

    normalized = normalize_text(model_name)

    for prefix, group_name in KNOWN_GROUP_PREFIXES.items():
        if normalized.startswith(prefix):
            return group_name

    return extract_auto_group(normalized)


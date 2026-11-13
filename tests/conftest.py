"""Keep deterministic tests offline even when a developer has proxy credentials in .env."""
import os

# config.py uses load_dotenv(..., override=False), so these explicit empty values
# prevent test collection from activating remote embeddings or chat calls.
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""
os.environ["OPENAI_MODEL"] = ""
os.environ["OPENAI_EMBEDDING_MODEL"] = ""

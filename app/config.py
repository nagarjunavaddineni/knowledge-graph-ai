import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    openai_api_key: str
    openai_model: str
    app_env: str
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

        settings.validate()
        return settings

    def validate(self) -> None:
        missing_variables = []

        required_values = {
            "NEO4J_URI": self.neo4j_uri,
            "NEO4J_PASSWORD": self.neo4j_password,
            "OPENAI_API_KEY": self.openai_api_key,
        }

        for variable_name, value in required_values.items():
            if not value:
                missing_variables.append(variable_name)

        if missing_variables:
            missing = ", ".join(missing_variables)
            raise ValueError(
                f"Missing required environment variables: {missing}"
            )


settings = Settings.from_environment()
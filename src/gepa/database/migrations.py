"""Database migration utilities."""

import asyncio
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncEngine
import structlog

from .connection import DatabaseManager
from .models import Base

logger = structlog.get_logger(__name__)


class MigrationManager:
    """Manages database migrations using Alembic."""
    
    def __init__(self, db_manager: DatabaseManager, alembic_cfg_path: Optional[Path] = None):
        self.db_manager = db_manager
        self.alembic_cfg_path = alembic_cfg_path or Path("alembic.ini")
    
    def _get_alembic_config(self) -> Config:
        """Get Alembic configuration."""
        alembic_cfg = Config(str(self.alembic_cfg_path))
        alembic_cfg.set_main_option("sqlalchemy.url", self.db_manager.config.url)
        return alembic_cfg
    
    async def init_alembic(self, directory: str = "migrations") -> None:
        """Initialize Alembic in the project."""
        alembic_cfg = self._get_alembic_config()
        command.init(alembic_cfg, directory)
        logger.info("Initialized Alembic", directory=directory)
    
    async def create_migration(self, message: str, autogenerate: bool = True) -> None:
        """Create a new migration."""
        alembic_cfg = self._get_alembic_config()
        command.revision(alembic_cfg, message=message, autogenerate=autogenerate)
        logger.info("Created migration", message=message)
    
    async def upgrade(self, revision: str = "head") -> None:
        """Upgrade database to a specific revision."""
        alembic_cfg = self._get_alembic_config()
        command.upgrade(alembic_cfg, revision)
        logger.info("Upgraded database", revision=revision)
    
    async def downgrade(self, revision: str = "-1") -> None:
        """Downgrade database to a specific revision."""
        alembic_cfg = self._get_alembic_config()
        command.downgrade(alembic_cfg, revision)
        logger.info("Downgraded database", revision=revision)
    
    async def get_current_revision(self) -> Optional[str]:
        """Get current database revision."""
        async with self.db_manager.engine.connect() as conn:
            context = MigrationContext.configure(conn.sync_connection)
            return context.get_current_revision()
    
    async def get_head_revision(self) -> Optional[str]:
        """Get head revision from scripts."""
        alembic_cfg = self._get_alembic_config()
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        return script_dir.get_current_head()
    
    async def is_up_to_date(self) -> bool:
        """Check if database is up to date."""
        current = await self.get_current_revision()
        head = await self.get_head_revision()
        return current == head


async def init_database(db_manager: DatabaseManager, create_tables: bool = True) -> None:
    """Initialize database with tables."""
    if create_tables:
        await db_manager.create_tables()
        logger.info("Database initialized with tables")
    
    # Check if migrations are needed
    try:
        migration_manager = MigrationManager(db_manager)
        is_up_to_date = await migration_manager.is_up_to_date()
        
        if not is_up_to_date:
            logger.warning(
                "Database schema may be out of date. Consider running migrations."
            )
    except Exception as e:
        logger.warning("Could not check migration status", error=str(e))


async def reset_database(db_manager: DatabaseManager) -> None:
    """Reset database by dropping and recreating all tables."""
    logger.warning("Resetting database - all data will be lost!")
    
    await db_manager.drop_tables()
    await db_manager.create_tables()
    
    logger.info("Database reset complete")


# CLI-friendly functions
async def create_migration_cli(
    db_url: str,
    message: str,
    alembic_cfg_path: Optional[str] = None
) -> None:
    """Create migration from CLI."""
    from ..config import DatabaseConfig
    
    config = DatabaseConfig(url=db_url)
    db_manager = DatabaseManager(config)
    
    migration_manager = MigrationManager(
        db_manager,
        Path(alembic_cfg_path) if alembic_cfg_path else None
    )
    
    await migration_manager.create_migration(message)


async def upgrade_database_cli(
    db_url: str,
    revision: str = "head",
    alembic_cfg_path: Optional[str] = None
) -> None:
    """Upgrade database from CLI."""
    from ..config import DatabaseConfig
    
    config = DatabaseConfig(url=db_url)
    db_manager = DatabaseManager(config)
    
    migration_manager = MigrationManager(
        db_manager,
        Path(alembic_cfg_path) if alembic_cfg_path else None
    )
    
    await migration_manager.upgrade(revision)


async def init_database_cli(
    db_url: str,
    reset: bool = False,
    create_tables: bool = True
) -> None:
    """Initialize database from CLI."""
    from ..config import DatabaseConfig
    
    config = DatabaseConfig(url=db_url)
    db_manager = DatabaseManager(config)
    
    try:
        if reset:
            await reset_database(db_manager)
        else:
            await init_database(db_manager, create_tables)
        
        logger.info("Database initialization complete")
        
    finally:
        await db_manager.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration utilities")
    parser.add_argument("--db-url", required=True, help="Database URL")
    parser.add_argument("command", choices=["init", "reset", "migrate", "upgrade"])
    parser.add_argument("--message", help="Migration message")
    parser.add_argument("--revision", default="head", help="Migration revision")
    parser.add_argument("--reset", action="store_true", help="Reset database")
    parser.add_argument("--alembic-cfg", help="Alembic config file path")
    
    args = parser.parse_args()
    
    if args.command == "init":
        asyncio.run(init_database_cli(args.db_url, args.reset))
    elif args.command == "reset":
        asyncio.run(init_database_cli(args.db_url, reset=True))
    elif args.command == "migrate":
        if not args.message:
            parser.error("--message required for migrate command")
        asyncio.run(create_migration_cli(args.db_url, args.message, args.alembic_cfg))
    elif args.command == "upgrade":
        asyncio.run(upgrade_database_cli(args.db_url, args.revision, args.alembic_cfg))
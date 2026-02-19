from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import Config


engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    Base.metadata.create_all(bind=engine)


from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker



SQLALCHEMY_DATABASE_URL_FOR_NEW_SPAM_MESSAGES = "sqlite:///./contact_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL_FOR_NEW_SPAM_MESSAGES, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Contact(Base):
    __tablename__ = "contacts" 

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    text_box = Column(String, nullable=False)




def init_db():
    Base.metadata.create_all(bind=engine)




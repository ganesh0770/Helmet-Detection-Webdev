from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL_FOR_CONTACT = "sqlite:///./contact_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL_FOR_CONTACT, connect_args={"check_same_thread": False}
)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Contact(Base):
    __tablename__ = "contacts_complaints" 

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    text_box = Column(String, nullable=False)





def init_db():
    Base.metadata.create_all(bind=engine)







from sqlalchemy.orm import Session
from datadb import SessionLocal, Contact,new_message

db: Session = SessionLocal()

contacts = db.query(Contact).all()
msg = db.query(new_message).all()
for c in contacts:
    print(c.id, c.name, c.email, c.text_box)
for c in msg:
    print(c.id, c.message, c.label)
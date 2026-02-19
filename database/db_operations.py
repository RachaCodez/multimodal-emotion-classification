from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database.db_config import Base
import bcrypt


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")

    def set_password(self, password: str):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    prediction_date = Column(DateTime, default=datetime.utcnow)
    input_type = Column(Enum('speech', 'text', 'image', 'multimodal'))
    predicted_emotion = Column(String(50))
    confidence_score = Column(Float)
    speech_emotion = Column(String(50))
    text_emotion = Column(String(50))
    image_emotion = Column(String(50))
    speech_confidence = Column(Float)
    text_confidence = Column(Float)
    image_confidence = Column(Float)
    file_path = Column(String(255))

    user = relationship("User", back_populates="predictions")


def create_user(session, username, email, password):
    user = User(username=username, email=email)
    user.set_password(password)
    session.add(user)
    session.commit()
    return user


def save_prediction(session, user_id, **kwargs):
    prediction = Prediction(user_id=user_id, **kwargs)
    session.add(prediction)
    session.commit()
    return prediction


def get_user_predictions(session, user_id):
    return session.query(Prediction).filter_by(user_id=user_id).order_by(Prediction.prediction_date.desc()).all()


# Additional analytics tables per specification
class EmotionStatistic(Base):
    __tablename__ = 'emotion_statistics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    emotion = Column(String(50), unique=True, nullable=False)
    count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    __tablename__ = 'model_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100))
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, default=datetime.utcnow)


def increment_emotion_stat(session, emotion: str):
    if not emotion:
        return
    stat = session.query(EmotionStatistic).filter_by(emotion=emotion).first()
    if not stat:
        stat = EmotionStatistic(emotion=emotion, count=1)
        session.add(stat)
    else:
        stat.count = (stat.count or 0) + 1
        stat.last_updated = datetime.utcnow()
    session.commit()


def get_emotion_statistics(session):
    return session.query(EmotionStatistic).all()


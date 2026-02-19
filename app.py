"""
Flask application for the Multimodal Emotion Classification System.

Features:
- Authentication (register, login, logout)
- Single modality predictions (speech, text, image)
- Multimodal predictions with weighted fusion fallback
- History & dashboard with simple charts

Note: Inference modules include safe fallbacks if models are not found.
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from config import Config
from database.db_config import SessionLocal, init_db
from database.db_operations import (
    create_user,
    get_user_predictions,
    save_prediction,
    User,
    Prediction,
    increment_emotion_stat,
    get_emotion_statistics,
)

# Inference modules
from inference.speech_inference import SpeechInference
from inference.text_inference import TextInference
from inference.image_inference import ImageInference
from inference.multimodal_fusion import MultimodalFusion
from logging_config import setup_logging
from security import sanitize_text, sanitize_filename, validate_email, validate_username, validate_password


load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

# Session Security Configuration
app.config.update(
    SESSION_COOKIE_SECURE=Config.SESSION_COOKIE_SECURE,
    SESSION_COOKIE_HTTPONLY=Config.SESSION_COOKIE_HTTPONLY,
    SESSION_COOKIE_SAMESITE=Config.SESSION_COOKIE_SAMESITE,
    PERMANENT_SESSION_LIFETIME=Config.PERMANENT_SESSION_LIFETIME,
    SESSION_REFRESH_EACH_REQUEST=Config.SESSION_REFRESH_EACH_REQUEST,
)

# CSRF Protection
try:
    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect(app)
except ImportError:
    print("Warning: Flask-WTF not installed. CSRF protection disabled.")
    csrf = None

# Rate Limiting
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
except ImportError:
    print("Warning: Flask-Limiter not installed. Rate limiting disabled.")
    limiter = None

# HTTPS Enforcement (optional, for production)
try:
    from flask_talisman import Talisman
    if os.environ.get('FLASK_ENV') == 'production':
        Talisman(app, force_https=True, content_security_policy=None)
except ImportError:
    print("Warning: Flask-Talisman not installed. HTTPS enforcement disabled.")

# Logging Configuration
logger = setup_logging(app)

# Security Headers Middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    for header, value in Config.SECURITY_HEADERS.items():
        response.headers[header] = value
    return response

# Expose Config in templates
app.jinja_env.globals['config'] = Config


def allowed_file(filename: str, kind: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if kind == 'audio':
        return ext in Config.ALLOWED_AUDIO_EXTENSIONS
    if kind == 'image':
        return ext in Config.ALLOWED_IMAGE_EXTENSIONS
    return False


def login_required(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return func(*args, **kwargs)

    return wrapper


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("3 per hour") if limiter else lambda f: f
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        db = SessionLocal()
        try:
            existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
            if existing:
                flash('Username or email already exists.', 'danger')
                return render_template('register.html')

            user = create_user(db, username, email, password)
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Registration successful. Welcome!', 'success')
            return redirect(url_for('dashboard'))
        finally:
            db.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per 15 minutes") if limiter else lambda f: f
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user or not user.check_password(password):
                flash('Invalid username or password.', 'danger')
                return render_template('login.html')
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        finally:
            db.close()

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    db = SessionLocal()
    try:
        user_id = session['user_id']
        preds = db.query(Prediction).filter(Prediction.user_id == user_id).order_by(Prediction.prediction_date.desc()).limit(5).all()
        total_count = db.query(Prediction).filter(Prediction.user_id == user_id).count()
        # Simple distribution summary
        distribution = {e: 0 for e in Config.EMOTIONS}
        for p in db.query(Prediction).filter(Prediction.user_id == user_id).all():
            if p.predicted_emotion in distribution:
                distribution[p.predicted_emotion] += 1
        # Most common
        most_common = max(distribution, key=distribution.get) if total_count > 0 else None
        
        chart_labels = list(distribution.keys())
        chart_values = [distribution[k] for k in chart_labels]
        return render_template(
            'dashboard.html',
            recent=preds,
            total_count=total_count,
            most_common=most_common,
            chart_labels=chart_labels,
            chart_values=chart_values,
        )
    finally:
        db.close()


@app.route('/predict/speech', methods=['GET', 'POST'])
@login_required
def predict_speech():
    if request.method == 'POST':
        audio_file = request.files.get('audio_file')
        if audio_file and allowed_file(audio_file.filename, 'audio'):
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            audio_file.save(filepath)

            predictor = SpeechInference()
            result = predictor.predict(filepath)

            db = SessionLocal()
            try:
                save_prediction(
                    db,
                    user_id=session.get('user_id'),
                    input_type='speech',
                    predicted_emotion=result['emotion'],
                    confidence_score=result['confidence'],
                    speech_emotion=result['emotion'],
                    speech_confidence=result['confidence'],
                    file_path=filepath,
                )
                increment_emotion_stat(db, result['emotion'])
            finally:
                db.close()

            return render_template('results.html', modality='speech', result=result)

        flash('Invalid audio file.', 'danger')
    return render_template('speech_input.html')


@app.route('/predict/text', methods=['GET', 'POST'])
@login_required
def predict_text():
    if request.method == 'POST':
        text_input = request.form.get('text_input')
        if text_input:
            predictor = TextInference()
            result = predictor.predict(text_input)

            db = SessionLocal()
            try:
                save_prediction(
                    db,
                    user_id=session.get('user_id'),
                    input_type='text',
                    predicted_emotion=result['emotion'],
                    confidence_score=result['confidence'],
                    text_emotion=result['emotion'],
                    text_confidence=result['confidence'],
                )
                increment_emotion_stat(db, result['emotion'])
            finally:
                db.close()

            return render_template('results.html', modality='text', result=result, text=text_input)

        flash('Please enter some text.', 'warning')
    return render_template('text_input.html')


@app.route('/predict/image', methods=['GET', 'POST'])
@login_required
def predict_image():
    if request.method == 'POST':
        image_file = request.files.get('image_file')
        if image_file and allowed_file(image_file.filename, 'image'):
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            image_file.save(filepath)

            predictor = ImageInference()
            result = predictor.predict(filepath)

            db = SessionLocal()
            try:
                save_prediction(
                    db,
                    user_id=session.get('user_id'),
                    input_type='image',
                    predicted_emotion=result['emotion'],
                    confidence_score=result['confidence'],
                    image_emotion=result['emotion'],
                    image_confidence=result['confidence'],
                    file_path=filepath,
                )
                increment_emotion_stat(db, result['emotion'])
            finally:
                db.close()

            return render_template('results.html', modality='image', result=result, image_path=filepath)

        flash('Invalid image file.', 'danger')
    return render_template('image_input.html')


@app.route('/predict/multimodal', methods=['GET', 'POST'])
@login_required
def predict_multimodal():
    if request.method == 'POST':
        audio_file = request.files.get('audio_file')
        text_input = request.form.get('text_input')
        image_file = request.files.get('image_file')

        audio_path = None
        image_path = None

        if audio_file and allowed_file(audio_file.filename, 'audio'):
            fname = secure_filename(audio_file.filename)
            audio_path = os.path.join(Config.UPLOAD_FOLDER, fname)
            audio_file.save(audio_path)

        if image_file and allowed_file(image_file.filename, 'image'):
            fname = secure_filename(image_file.filename)
            image_path = os.path.join(Config.UPLOAD_FOLDER, fname)
            image_file.save(image_path)

        fusion = MultimodalFusion()
        results = fusion.predict_multimodal(audio_path, text_input, image_path)

        db = SessionLocal()
        try:
            save_prediction(
                db,
                user_id=session.get('user_id'),
                input_type='multimodal',
                predicted_emotion=(results.get('fusion') or results.get('speech') or results.get('text') or results.get('image') or {}).get('emotion'),
                confidence_score=(results.get('fusion') or results.get('speech') or results.get('text') or results.get('image') or {}).get('confidence'),
                speech_emotion=results.get('speech', {}).get('emotion'),
                text_emotion=results.get('text', {}).get('emotion'),
                image_emotion=results.get('image', {}).get('emotion'),
                speech_confidence=results.get('speech', {}).get('confidence'),
                text_confidence=results.get('text', {}).get('confidence'),
                image_confidence=results.get('image', {}).get('confidence'),
            )
            fusion_label = (results.get('fusion') or {}).get('emotion')
            if fusion_label:
                increment_emotion_stat(db, fusion_label)
        finally:
            db.close()

        return render_template('results.html', modality='multimodal', result=results, image_path=image_path, text=text_input)

    return render_template('multimodal_input.html')


@app.route('/history')
@login_required
def history():
    db = SessionLocal()
    try:
        # Optional filters
        emotion = request.args.get('emotion')
        modality = request.args.get('modality')
        start = request.args.get('start')
        end = request.args.get('end')

        q = db.query(Prediction).filter(Prediction.user_id == session['user_id'])
        if emotion:
            q = q.filter(Prediction.predicted_emotion == emotion)
        if modality:
            q = q.filter(Prediction.input_type == modality)
        if start:
            try:
                from datetime import datetime as _dt
                q = q.filter(Prediction.prediction_date >= _dt.fromisoformat(start))
            except Exception:
                pass
        if end:
            try:
                from datetime import datetime as _dt
                q = q.filter(Prediction.prediction_date <= _dt.fromisoformat(end))
            except Exception:
                pass
        preds = q.order_by(Prediction.prediction_date.desc()).all()
        return render_template('history.html', predictions=preds)
    finally:
        db.close()


@app.route('/history/export.csv')
@login_required
def export_history_csv():
    import io, csv
    db = SessionLocal()
    try:
        preds = db.query(Prediction).filter(Prediction.user_id == session['user_id']).order_by(Prediction.prediction_date.desc()).all()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['date','modality','emotion','confidence','speech_emotion','text_emotion','image_emotion'])
        for p in preds:
            writer.writerow([
                p.prediction_date.isoformat(sep=' ', timespec='seconds'),
                p.input_type,
                p.predicted_emotion,
                f"{(p.confidence_score or 0):.4f}",
                p.speech_emotion or '',
                p.text_emotion or '',
                p.image_emotion or '',
            ])
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='history.csv')
    finally:
        db.close()


@app.route('/statistics')
@login_required
def statistics_page():
    db = SessionLocal()
    try:
        stats = get_emotion_statistics(db)
        labels = [s.emotion for s in stats]
        values = [s.count for s in stats]
        return render_template('statistics.html', labels=labels, values=values)
    finally:
        db.close()


# API endpoints
@app.post('/api/register')
def api_register():
    data = request.json or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not (username and email and password):
        return jsonify({'error': 'username, email, and password are required'}), 400
    db = SessionLocal()
    try:
        existing = db.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing:
            return jsonify({'error': 'username or email exists'}), 409
        user = create_user(db, username, email, password)
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'id': user.id, 'username': user.username, 'email': user.email}), 201
    finally:
        db.close()


@app.post('/api/login')
def api_login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')
    if not (username and password):
        return jsonify({'error': 'username and password required'}), 400
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'invalid credentials'}), 401
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'message': 'logged in', 'username': user.username})
    finally:
        db.close()


@app.post('/api/logout')
def api_logout():
    session.clear()
    return jsonify({'message': 'logged out'})


@app.get('/api/user/profile')
def api_user_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    return jsonify({'id': session['user_id'], 'username': session.get('username')})


@app.post('/api/predict/speech')
def api_predict_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'multipart/form-data with audio file required (field name: audio)'}), 400
    f = request.files['audio']
    if not (f and allowed_file(f.filename, 'audio')):
        return jsonify({'error': 'invalid file'}), 400
    fname = secure_filename(f.filename)
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    f.save(path)
    result = SpeechInference().predict(path)
    if 'user_id' in session:
        db = SessionLocal()
        try:
            save_prediction(db, user_id=session['user_id'], input_type='speech', predicted_emotion=result['emotion'], confidence_score=result['confidence'], speech_emotion=result['emotion'], speech_confidence=result['confidence'], file_path=path)
            increment_emotion_stat(db, result['emotion'])
        finally:
            db.close()
    return jsonify(result)


@app.post('/api/predict/text')
def api_predict_text():
    data = request.json or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'text is required'}), 400
    result = TextInference().predict(text)
    if 'user_id' in session:
        db = SessionLocal()
        try:
            save_prediction(db, user_id=session['user_id'], input_type='text', predicted_emotion=result['emotion'], confidence_score=result['confidence'], text_emotion=result['emotion'], text_confidence=result['confidence'])
            increment_emotion_stat(db, result['emotion'])
        finally:
            db.close()
    return jsonify(result)


@app.post('/api/predict/image')
def api_predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'multipart/form-data with image file required (field name: image)'}), 400
    f = request.files['image']
    if not (f and allowed_file(f.filename, 'image')):
        return jsonify({'error': 'invalid file'}), 400
    fname = secure_filename(f.filename)
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    f.save(path)
    result = ImageInference().predict(path)
    if 'user_id' in session:
        db = SessionLocal()
        try:
            save_prediction(db, user_id=session['user_id'], input_type='image', predicted_emotion=result['emotion'], confidence_score=result['confidence'], image_emotion=result['emotion'], image_confidence=result['confidence'], file_path=path)
            increment_emotion_stat(db, result['emotion'])
        finally:
            db.close()
    return jsonify(result)


@app.post('/api/predict/multimodal')
def api_predict_multimodal():
    text = request.form.get('text') or (request.json or {}).get('text')
    audio_f = request.files.get('audio') if 'audio' in request.files else None
    image_f = request.files.get('image') if 'image' in request.files else None

    audio_path = None
    image_path = None
    if audio_f and allowed_file(audio_f.filename, 'audio'):
        afn = secure_filename(audio_f.filename)
        audio_path = os.path.join(Config.UPLOAD_FOLDER, afn)
        audio_f.save(audio_path)
    if image_f and allowed_file(image_f.filename, 'image'):
        ifn = secure_filename(image_f.filename)
        image_path = os.path.join(Config.UPLOAD_FOLDER, ifn)
        image_f.save(image_path)

    results = MultimodalFusion().predict_multimodal(audio_path, text, image_path)
    if 'user_id' in session:
        db = SessionLocal()
        try:
            save_prediction(db, user_id=session['user_id'], input_type='multimodal', predicted_emotion=(results.get('fusion') or results.get('speech') or results.get('text') or results.get('image') or {}).get('emotion'), confidence_score=(results.get('fusion') or results.get('speech') or results.get('text') or results.get('image') or {}).get('confidence'), speech_emotion=results.get('speech', {}).get('emotion'), text_emotion=results.get('text', {}).get('emotion'), image_emotion=results.get('image', {}).get('emotion'), speech_confidence=results.get('speech', {}).get('confidence'), text_confidence=results.get('text', {}).get('confidence'), image_confidence=results.get('image', {}).get('confidence'))
            fusion_label = (results.get('fusion') or {}).get('emotion')
            if fusion_label:
                increment_emotion_stat(db, fusion_label)
        finally:
            db.close()
    return jsonify(results)


@app.get('/api/predictions')
def api_predictions():
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    db = SessionLocal()
    try:
        preds = get_user_predictions(db, session['user_id'])
        data = [
            {
                'id': p.id,
                'date': p.prediction_date.isoformat(sep=' ', timespec='seconds'),
                'modality': p.input_type,
                'emotion': p.predicted_emotion,
                'confidence': p.confidence_score,
            } for p in preds
        ]
        return jsonify(data)
    finally:
        db.close()


@app.delete('/api/predictions/<int:pid>')
def api_delete_prediction(pid: int):
    if 'user_id' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    db = SessionLocal()
    try:
        p = db.get(Prediction, pid)
        if not p or p.user_id != session['user_id']:
            return jsonify({'error': 'not found'}), 404
        db.delete(p)
        db.commit()
        return jsonify({'message': 'deleted'})
    finally:
        db.close()


@app.get('/api/statistics')
def api_statistics():
    db = SessionLocal()
    try:
        stats = get_emotion_statistics(db)
        return jsonify([{ 'emotion': s.emotion, 'count': s.count } for s in stats])
    finally:
        db.close()


if __name__ == '__main__':
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    with app.app_context():
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return (render_template('index.html',), 413)

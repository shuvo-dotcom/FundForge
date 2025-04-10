from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)
    
    # Ensure static/plots directory exists
    os.makedirs(os.path.join(app.static_folder, 'plots'), exist_ok=True)
    
    # Register blueprint
    from app.routes.index_fund import index_fund_bp
    app.register_blueprint(index_fund_bp)
    
    return app 
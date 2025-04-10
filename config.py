import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change'
    
    # Analysis settings
    MIN_STOCKS = int(os.environ.get('MIN_STOCKS', 10))
    MAX_STOCKS = int(os.environ.get('MAX_STOCKS', 100))
    DEFAULT_STOCKS = int(os.environ.get('DEFAULT_STOCKS', 20))
    DEFAULT_PERIOD = os.environ.get('DEFAULT_PERIOD', '3mo')
    
    # Flask
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'postgresql://localhost/fundforge'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Blockchain
    POLYGON_RPC_URL = os.environ.get('POLYGON_RPC_URL') or \
        'https://polygon-mumbai.infura.io/v3/YOUR-PROJECT-ID'
    POLYGON_CHAIN_ID = int(os.environ.get('POLYGON_CHAIN_ID', 80001))  # Mumbai testnet
    
    # Web3 Storage
    WEB3_STORAGE_TOKEN = os.environ.get('WEB3_STORAGE_TOKEN')
    
    # Application Settings
    CAMPAIGNS_PER_PAGE = 12
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    
    # File upload configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/uploads')
    
    # Web3 configuration
    WEB3_PROVIDER_URI = os.environ.get('WEB3_PROVIDER_URI') or 'http://localhost:8545'
    
    # Smart contract configuration
    CONTRACT_ADDRESS = os.environ.get('CONTRACT_ADDRESS')
    CONTRACT_ABI = os.environ.get('CONTRACT_ABI') 
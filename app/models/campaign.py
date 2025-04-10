from datetime import datetime
from app import db

class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    goal_amount = db.Column(db.Numeric(10, 2), nullable=False)
    current_amount = db.Column(db.Numeric(10, 2), default=0)
    start_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=False)
    image_url = db.Column(db.String(255))
    contract_address = db.Column(db.String(42), unique=True)
    status = db.Column(db.String(20), default='active')  # active, completed, expired
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    donations = db.relationship('Donation', backref='campaign', lazy='dynamic')

    def __repr__(self):
        return f'<Campaign {self.title}>'

    @property
    def progress_percentage(self):
        if float(self.goal_amount) == 0:
            return 0
        return min(100, (float(self.current_amount) / float(self.goal_amount)) * 100)

    @property
    def is_expired(self):
        return datetime.utcnow() > self.end_date

    @property
    def is_completed(self):
        return float(self.current_amount) >= float(self.goal_amount) 
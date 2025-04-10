from datetime import datetime
from app import db

class Donation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    transaction_hash = db.Column(db.String(66), unique=True)
    message = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    donor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    campaign_id = db.Column(db.Integer, db.ForeignKey('campaign.id'), nullable=False)

    def __repr__(self):
        return f'<Donation {self.amount} to Campaign {self.campaign_id}>' 
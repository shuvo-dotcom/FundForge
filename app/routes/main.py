from flask import Blueprint, render_template, request
from app.models import Campaign
from app import db
from sqlalchemy import desc

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 9  # Show 9 campaigns per page
    
    # Get active campaigns ordered by creation date
    campaigns = Campaign.query.filter(
        Campaign.end_date > db.func.now(),
        Campaign.current_amount < Campaign.goal_amount
    ).order_by(desc(Campaign.created_at)).paginate(page=page, per_page=per_page)
    
    return render_template('index.html', campaigns=campaigns)

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html') 
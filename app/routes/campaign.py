from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app.models import Campaign, Donation
from app import db
from datetime import datetime
import os
from werkzeug.utils import secure_filename

# Create the blueprint
campaign_bp = Blueprint('campaign', __name__)

@campaign_bp.route('/create', methods=['GET', 'POST'])
@login_required
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        goal_amount = float(request.form.get('goal_amount'))
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
        
        # Handle image upload
        image = request.files.get('image')
        image_filename = None
        if image:
            filename = secure_filename(image.filename)
            image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image_filename)
            image.save(image_path)
        
        campaign = Campaign(
            title=title,
            description=description,
            goal_amount=goal_amount,
            end_date=end_date,
            image_filename=image_filename,
            creator_id=current_user.id
        )
        
        db.session.add(campaign)
        db.session.commit()
        
        flash('Campaign created successfully!', 'success')
        return redirect(url_for('campaign.view', id=campaign.id))
    
    return render_template('campaign/create.html')

@campaign_bp.route('/<int:id>')
def view(id):
    campaign = Campaign.query.get_or_404(id)
    return render_template('campaign/view.html', campaign=campaign)

@campaign_bp.route('/<int:id>/donate', methods=['POST'])
@login_required
def donate(id):
    campaign = Campaign.query.get_or_404(id)
    
    if campaign.is_expired:
        flash('This campaign has expired.', 'error')
        return redirect(url_for('campaign.view', id=id))
    
    if campaign.is_completed:
        flash('This campaign has already reached its goal.', 'error')
        return redirect(url_for('campaign.view', id=id))
    
    amount = float(request.form.get('amount'))
    
    donation = Donation(
        amount=amount,
        campaign_id=id,
        donor_id=current_user.id
    )
    
    db.session.add(donation)
    db.session.commit()
    
    flash(f'Thank you for your donation of {amount} ETH!', 'success')
    return redirect(url_for('campaign.view', id=id))

@campaign_bp.route('/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit(id):
    campaign = Campaign.query.get_or_404(id)
    
    if campaign.creator_id != current_user.id:
        flash('You are not authorized to edit this campaign.', 'error')
        return redirect(url_for('campaign.view', id=id))
    
    if request.method == 'POST':
        campaign.title = request.form.get('title')
        campaign.description = request.form.get('description')
        campaign.goal_amount = float(request.form.get('goal_amount'))
        campaign.end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
        
        # Handle image upload
        image = request.files.get('image')
        if image:
            # Delete old image if exists
            if campaign.image_filename:
                old_image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], campaign.image_filename)
                if os.path.exists(old_image_path):
                    os.remove(old_image_path)
            
            filename = secure_filename(image.filename)
            campaign.image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            image_path = os.path.join(current_app.config['UPLOAD_FOLDER'], campaign.image_filename)
            image.save(image_path)
        
        db.session.commit()
        flash('Campaign updated successfully!', 'success')
        return redirect(url_for('campaign.view', id=id))
    
    return render_template('campaign/edit.html', campaign=campaign)

@campaign_bp.route('/my-campaigns')
@login_required
def my_campaigns():
    campaigns = Campaign.query.filter_by(creator_id=current_user.id).all()
    return render_template('campaign/my_campaigns.html', campaigns=campaigns) 
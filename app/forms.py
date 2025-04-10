from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, DecimalField, DateField, FileField
from wtforms.validators import DataRequired, Length, NumberRange, Optional
from datetime import datetime

class CampaignForm(FlaskForm):
    title = StringField('Title', validators=[
        DataRequired(),
        Length(min=3, max=100, message='Title must be between 3 and 100 characters')
    ])
    description = TextAreaField('Description', validators=[
        DataRequired(),
        Length(min=10, max=2000, message='Description must be between 10 and 2000 characters')
    ])
    goal_amount = DecimalField('Goal Amount (ETH)', validators=[
        DataRequired(),
        NumberRange(min=0.001, message='Minimum goal amount is 0.001 ETH')
    ])
    end_date = DateField('End Date', validators=[
        DataRequired()
    ])
    image = FileField('Campaign Image', validators=[
        Optional()
    ])

    def validate_end_date(self, field):
        if field.data < datetime.now().date():
            raise ValueError('End date must be in the future') 
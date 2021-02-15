from flask_wtf import FlaskForm
from wtforms import SubmitField, HiddenField


class SelectImageForm(FlaskForm):
    index = HiddenField()
    selected_feature = SubmitField()

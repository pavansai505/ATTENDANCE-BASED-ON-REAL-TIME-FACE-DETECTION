from flask import Flask, render_template, request, url_for, session, redirect
from pymongo import MongoClient
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField
from wtforms.validators import InputRequired, Length
import math

app = Flask(__name__)
app.secret_key = 'any randon key'
# CONNECTION
client = MongoClient('mongodb://localhost:27017')
db = client.AAS
student = db.student_details
register_db = db.registration_form
attendance = db.attendance
# FILE STORAGE
UPLOAD_FOLDER = 'C:/AAS/phase_1/static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class RegistrationForm(FlaskForm):
    user_name = StringField('username', validators=[InputRequired()])
    mail_id = StringField('mail_id', validators=[InputRequired()])
    phone_no = IntegerField('phone', validators=[InputRequired()])
    roll_no = StringField('roll_no', validators=[InputRequired(), Length(min=10, max=10)])


class LoginForm(FlaskForm):
    roll_no = StringField('roll_no', validators=[InputRequired(), Length(min=10, max=10)])
    password = PasswordField('password', validators=[InputRequired()])


@app.route('/')
def start():
    form = LoginForm()
    return render_template('login.html', form=form)


@app.route('/login', methods=['POST', 'GET'])
def first():
    form = LoginForm()
    if form.validate_on_submit():
        roll_no = form.roll_no.data
        password = form.password.data
        student_find = student.find_one({'roll_no': roll_no})
        if (student_find):
            if (password == student_find['password'] and student_find['type'] == 'admin'):
                session['roll'] = roll_no
                students = student.find({}).limit(60)

                return render_template('dashboard.html', students=students)
            else:
                session['roll'] = roll_no
                attendance_find = attendance.find({'roll_no': roll_no}).sort("_id", -1)
                for_list = attendance.find_one({'roll_no': roll_no})
                lst = list(for_list.keys())
                lst = lst[4:]
                perc = 0
                for j in lst:
                    perc = perc + count(j, attendance_find) * 100
                perc = perc // len(lst)
                # print(attendance_find)
                return render_template('attendance.html', data_v=attendance_find, per=perc, num=roll_no)
        else:
            return 'id not_Found'
    else:
        return render_template('login.html', form=form)


def count(h, a):
    cop = 0
    coa = 0
    for i in a:
        if i[h] == 'P':
            cop = cop + 1
        coa = coa + 1
        return cop / coa


@app.route('/home_page')
def home_page():
    students = student.find({}).limit(60)

    return render_template('dashboard.html', students=students)


@app.route('/attendance_page', methods=['POST', 'GET'])
def attend():
    roll_no = request.form.get('roll')
    # roll_no=session['roll']
    attendance_find = attendance.find({'roll_no': roll_no}).sort('_id', -1)
    for_list = attendance.find_one({'roll_no': roll_no})
    lst = list(for_list.keys())
    lst = lst[4:]
    perc = 0
    for j in lst:
        perc = perc + count(j, attendance_find) * 100
    perc = round(perc / len(lst))
    return render_template('attendance.html', data_v=attendance_find, per=perc, num=roll_no)


def count(h, a):
    cop = 0
    coa = 0
    for i in a:
        if i[h] == 'P':
            cop = cop + 1
        coa = coa + 1
        return (cop / coa)


if __name__ == '_main_':
    app.run(debug=True)
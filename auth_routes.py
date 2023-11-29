import csv
from flask import Blueprint, render_template, request, redirect, url_for, flash, session

from werkzeug.security import check_password_hash
from models import db,  Users


auth_bp = Blueprint('auth', __name__)


def read_mobile_codes_from_csv():
    mobile_codes = []
    with open('mobile_code.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mobile_codes.append(row)
    return mobile_codes


# mobile_codes = [
#     {"Series": "123", "Operator": "Operator1", "Circle": "Circle1"},
#     {"Series": "456", "Operator": "Operator2", "Circle": "Circle2"},
#     # ... other data
# ]
# Load mobile codes from the CSV file
mobile_codes = read_mobile_codes_from_csv()

# @auth_bp.route('/logout')
# def logout():
#     session.pop('email', None)
#     user= session["user"]
#     if user:
#         id=user.id
#         update_user_logout_session(id)
#     return redirect('/login')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        fullname = request.form['fullname']

        password = request.form['password']
        mobile_no = request.form['mobile_no']  # Get mobile number from form
        address = request.form['address']      # Get address from form
        zipcode = request.form['zipcode']      # Get address from form

        # is_icici_demat_account=request.form['is_icici_demat_account']
        is_icici_demat_account = request.form.get(
            'is_icici_demat_account') == 'on'  # Convert checkbox value to boolean
        terms_and_conditions = request.form.get(
            'terms_and_conditions') == 'on'  # Convert checkbox value to boolean

        if not email or not password or not mobile_no or not address:
            return render_template('register.html', error="Dont enter Blank Email or Blank Password or Blank Mobile or Blank Address")
        # Save user data to SQLite database
        if Users.query.filter_by(email=email).first():
            return render_template('register.html', error="User with email:"+email + " already exists")

        if terms_and_conditions == False:
            return render_template('register.html', error="Please read and accept Terms and Conditions")

        # if is_icici_demat_account is None:
        #     is_icici_demat_account=False
        # else:
        #     is_icici_demat_account = True
        new_user = Users(email=email.upper(), password=password, mobile_no=mobile_no,
                         address=address, fullname=fullname, is_icici_demat_account=is_icici_demat_account, zipcode=zipcode)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html', mobile_codes=mobile_codes)

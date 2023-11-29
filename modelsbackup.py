from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RealTimeStockData(db.Model):
    __tablename__ = 'real_time_stock_data'
    id = Column(Integer, primary_key=True)
    datetime = Column(DateTime) 
    ticker_symbol = Column(String)
    volume = Column(Integer)
    ema = Column(Float)
    short_sma = Column(Float)
    long_sma = Column(Float)
    vwap = Column(Float)
    rsi = Column(Float) 
    high = Column(Float)
    low = Column(Float)
    # close_average = Column(Float)
    # average_open = Column(Float)
    # close_1d_ago = Column(Float)
    # close_2d_ago = Column(Float)
    # close_3d_ago = Column(Float)
    # close_4d_ago = Column(Float)
    # close_5d_ago = Column(Float)
    # open_1d_ago = Column(Float)
    # open_2d_ago = Column(Float)
    # open_3d_ago = Column(Float)
    # open_4d_ago = Column(Float)
    # open_5d_ago = Column(Float)
    # rise_1d_ago = Column(Integer)
    # rise_2d_ago = Column(Integer)
    # rise_3d_ago = Column(Integer)
    # rise_4d_ago = Column(Integer)
    # rise_5d_ago = Column(Integer)
    # rise_count = Column(Integer)
    # last_d_close = Column(Float)
    # last_d_open = Column(Float)
    start_earning= Column(Float)  
    first_close= Column(Float) 
    first_open = Column(Float)
    open = Column(Float)
    close = Column(Float)
    buy_signal = Column(String)
    sell_signal = Column(String)
    first_earning = Column(Float)
    earning_current = Column(Float)
    create_date_time = db.Column(db.String(100))

        


class MasterStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    company_name = db.Column(db.String(100), nullable=False)
    icici_symbol = db.Column(db.String(10), nullable=False)

    
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    mobile_no = db.Column(db.String(20), nullable=False)  # Set nullable=False
    address = db.Column(db.String(255), nullable=False)   # Set nullable=False

    # Define a one-to-many relationship with StockData
    stock_data = db.relationship('StockData', backref='user', lazy=True)


class MasterStockSelection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Date = db.Column(db.String(10))
    Ticker_Symbol = db.Column(db.String(10))
    Volume = db.Column(db.Integer)
    High = db.Column(db.Float)
    Low = db.Column(db.Float)
    Last_Day_Close = db.Column(db.Float)
    Open = db.Column(db.Float)
    Close = db.Column(db.Float)
    Close_1d_ago = db.Column(db.Float)
    Close_2d_ago = db.Column(db.Float)
    Close_3d_ago = db.Column(db.Float)
    Close_4d_ago = db.Column(db.Float)
    Close_5d_ago = db.Column(db.Float)
    Open_1d_ago = db.Column(db.Float)
    Open_2d_ago = db.Column(db.Float)
    Open_3d_ago = db.Column(db.Float)
    Open_4d_ago = db.Column(db.Float)
    Open_5d_ago = db.Column(db.Float)
    Rise_1d_ago = db.Column(db.Boolean)
    Rise_2d_ago = db.Column(db.Boolean)
    Rise_3d_ago = db.Column(db.Boolean)
    Rise_4d_ago = db.Column(db.Boolean)
    Rise_5d_ago = db.Column(db.Boolean)
    Rise_count = db.Column(db.Integer)
    Earnings = db.Column(db.Float)
    create_date_time = db.Column(db.String(100))
 
class StockData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date_time = db.Column(db.String(100), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    name_of_company = db.Column(db.String(100), nullable=False)
    rate = db.Column(db.Float, nullable=True)
     
    quantum = db.Column(db.Float, nullable=True)
    total_value=db.Column(db.Float, nullable=True)
    remark=db.Column(db.String(100), nullable=False)
    signal=db.Column(db.String(100), nullable=False)

    average_value=db.Column(db.Float, nullable=True)
    buy_amt=db.Column(db.Float, nullable=True)
    sell_amt=db.Column(db.Float, nullable=True)
    balance=db.Column(db.Float, nullable=True)







    # def __repr__(self):
    #     return f"StockData(id={self.id}, symbol='{self.symbol}', volume={self.volume}, " \
    #            f"high={self.high}, low={self.low}, rise_count={self.rise_count}, " \
    #            f"morning_open_value={self.morning_open_value}, " \
    #            f"current_open_value={self.current_open_value})"
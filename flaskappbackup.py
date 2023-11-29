import pandas_ta as ta

import io
import json
import concurrent.futures
import time
import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta    
import os
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask import session,Response,make_response
from sqlalchemy import MetaData, false, text, create_engine
from models import MasterStockSelection, RealTimeStockData, StockData, db, MasterStock, Users
import csv
from sqlalchemy.orm import sessionmaker
from auth_routes import auth_bp   
from apscheduler.schedulers.background import BackgroundScheduler
import sqlalchemy as sa

session = {}

basedir = os.path.abspath(os.path.dirname(__file__))
print("basedir:" + basedir)
app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    basedir, "masterstocks.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "my_secret_key"
db.init_app(app)
pd.options.mode.chained_assignment =  None 

email=''
with app.app_context():
    db.create_all()

app.register_blueprint(auth_bp)
engine = sa.create_engine("sqlite:///masterstocks.db", connect_args={"timeout": 600})
metadata = MetaData()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
)

# create a file handler that writes log messages to a file
file_handler = logging.FileHandler(os.path.join(basedir, "flaskapp.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
)

# add the file handler to the root logger
logging.getLogger().addHandler(file_handler)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

scheduler = BackgroundScheduler(daemon=True)

masterstocksdata=[]

@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if not email or not password:
            return render_template('login.html', error='Dont enter blank email or blank password')
        
        user = Users.query.filter_by(email=email.upper()).first()
        # print(user)
        # if not user or not check_password_hash(user.password, password):
        if not user or not (user.password == password):
            return render_template('login.html', error='Invalid email or password')
        session["email"] = email.upper()
        
        if user:
        # Convert the user object to a dictionary before storing it in the session
            session["user"] = {
            'id': user.id,
            'email': user.email,
            'is_admin': user.is_admin
            }
       
         
        return redirect('dashboard')
  else:
    return render_template('login.html')

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response



columns = [
    "Date",
    "Ticker_Symbol",
    "Volume",
    "ema",
    "short_sma",
    "long_sma",
    "vwap",
    "High",
    "Low",
    "Last_Day_Close",
    "Open",
    "Close",
    "Close_1d_ago",
    "Close_2d_ago",
    "Close_3d_ago",
    "Close_4d_ago",
    "Close_5d_ago",
    "Open_1d_ago",
    "Open_2d_ago",
    "Open_3d_ago",
    "Open_4d_ago",
    "Open_5d_ago",
    "Rise_1d_ago",
    "Rise_2d_ago",
    "Rise_3d_ago",
    "Rise_4d_ago",
    "Rise_5d_ago",
    "Rise_count",
    "Earnings",
    "create_date_time",
]

column_order = columns


def delete_realtime_stock_selections():
    with app.app_context():
        try:
            rows_deleted = db.session.query(RealTimeStockData).delete()
            db.session.commit()
            logging.info(
                f"Deleted {rows_deleted} rows from the RealTimeStockData model."
            )
        except Exception as e:
            logging.error(
                "An error occurred while loading the data into the database: {}".format(
                    str(e)
                )
            )
            print("Error occurred while writing data to database:", e)
            # raise e
        # finally:
        # engine.dispose()


def delete_stock_selections(tablename):
    with app.app_context():
        try:
            if tablename == "master_stock_selection":
                rows_deleted = db.session.query(MasterStockSelection).delete()
                db.session.commit()
                print('Row Deleted from table '+tablename)

            elif tablename == "real_time_stock_data":
                rows_deleted = db.session.query(RealTimeStockData).delete()
                db.session.commit()

            logging.info(
                f"Deleted {rows_deleted} rows from the " + tablename + " model."
            )
        except Exception as e:
            logging.error(
                "An error occurred while loading the data into the database: {}".format(
                    str(e)
                )
            )
            print("Error occurred while writing data to database:", e)
            # raise e
        finally:
            engine.dispose()


def write_stock_selections(table_name, latestdata):
    with app.app_context():
        try:
            logging.info(latestdata )
            print(latestdata.columns)
            records = latestdata.to_records(index=False)
            print(records)
            
            if table_name=='master_stock_selection':
                for index, row in latestdata.iterrows():
                    delete_date = row['Date']
                    delete_ticker_symbol =row['Ticker_Symbol']

# Execute the delete query using SQLAlchemy
                    rows_deleted = db.session.query(MasterStockSelection).filter( MasterStockSelection.Date == delete_date, MasterStockSelection.Ticker_Symbol == delete_ticker_symbol ).delete()
                    db.session.commit()
            else:
                for index, row in latestdata.iterrows():
                    delete_ticker_symbol =row['Ticker_Symbol']
                    delete_date = row['datetime']
                    rows_deleted = db.session.query(RealTimeStockData).filter( RealTimeStockData.datetime == delete_date, RealTimeStockData.ticker_symbol == delete_ticker_symbol ).delete()
                    db.session.commit()

            latestdata.to_sql(table_name, engine, if_exists="append", index=false)
            print("Data Written to Table")
            logging.info("Data loaded successfully into the database.")
            engine.dispose()
        except Exception as e:
            logging.error(
                "An error occurred while loading the data into the database: {}".format(
                    str(e)
                )
            )
            print("Error occurred while writing data to database:", e)
            raise e
        finally:
            engine.dispose()

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data["Adj Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.round(2) 

def get_latest_price(mode):
    logger = logging.getLogger("StockSelectionProgram")
    logger.setLevel(logging.INFO)

    # Define the log file
    log_file = "stock_selection.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Record the start time
    start_time = time.time()
    delete_stock_selections(mode)
    
    master_stocks=[]
    if mode == "master_stock_selection":
        master_stocks = get_master_stocks()
    else:
        master_stocks = get_stocks_selection()

    ticker_symbols = master_stocks
    print("Total Symbols in MasterStocks :", len(ticker_symbols))
    if len(ticker_symbols) <= 0:
        quit()

    # Calculate the date range
    end_date = (datetime.now() + timedelta(days=-0)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1 * 1)).strftime("%Y-%m-%d")
    print(end_date, start_date)

    # Set the interval
    interval = ""
    if mode == "master_stock_selection":
        interval = "1d"  # Daily data
    else:
        interval = "5m"
        # ticker_symbols=['COCHINSHIP.NS','ICICIBANK.NS']
    # Fetch historical data
    finaldata = pd.DataFrame()
    counter = 0
     
    for symbol in ticker_symbols:
        # stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval, period='1mo')
        if mode =="master_stock_selection":
            end_date = datetime.now()
            start_date = end_date - timedelta(days=15)
            print(end_date, start_date)
            # stock_data = yf.download(symbol, start=start_date, end=end_date)
            stock_data = yf.download(symbol,  start=start_date, end=end_date,interval=interval,group_by="ticker", prepost=True, proxy=None, rounding=True,  auto_adjust=False, back_adjust=False,actions=True, threads=True, timeout=None)
        else:
            #stock_data = yf.download(symbol,period="1mo",interval=interval,group_by="ticker", prepost=True, proxy=None, rounding=True,  auto_adjust=False, back_adjust=False,actions=True, threads=True, timeout=None)
            try:
                stock_data = yf.download(symbol,period="15d",interval=interval,group_by="ticker", prepost=True, proxy=None, rounding=True,  auto_adjust=False, back_adjust=False,actions=True, threads=True, timeout=None)                
    # Process or manipulate stock_data here
            except Exception as e:
                print("An error occurred:", e)
    
            # Add the Ticker_Symbol as a new column
        stock_data["Ticker_Symbol"] = symbol
        stock_data.reset_index(inplace=True)
        counter = counter + 1
        data = stock_data
        # print('Data printing')
        # print(data)
        if not stock_data.empty and mode != "master_stock_selection":
            today = datetime.now()
            # yesterday = today - timedelta(days=0)
            # yesterday_str= yesterday.strftime("%Y-%m-%d")
            # data = data[(data['Datetime'] >= yesterday_str)  ] 
            data['Date']=data['Datetime']
             # Calculate EMA
            ema_period = 3  # EMA period
            data['ema'] = data['Close'].transform(lambda x: x.ewm(span=ema_period, adjust=False).mean())

            # Calculate Short SMA
            short_sma_period = 10  # Short SMA period
            data['short_sma'] = data['Close'].transform(lambda x: x.rolling(window=short_sma_period).mean())

            # Calculate Long SMA
            long_sma_period = 50  # Long SMA period
            data['long_sma'] = data['Close'].transform(lambda x: x.rolling(window=long_sma_period).mean())
        
            yesterday = today - timedelta(days=0)
            yesterday_str= yesterday.strftime("%Y-%m-%d")
            data = data[(data['Datetime'] >= yesterday_str)  ]
           
            # Calculate the typical price (average of high, low, and close)
            data['TypicalPrice'] = (data['High'] + data['Low'] + data['Close']) / 3.0

            # Calculate the product of typical price and volume
            data['TypicalPriceVolume'] = data['TypicalPrice'] * data['Volume']

            # Calculate the cumulative sum of the product
            data['CumulativeTypicalPriceVolume'] = data['TypicalPriceVolume'].cumsum()

            # Calculate the cumulative sum of volume
            data['CumulativeVolume'] = data['Volume'].cumsum()

            # Calculate VWAP
            data['vwap'] = data['CumulativeTypicalPriceVolume'] / data['CumulativeVolume']
            data.set_index(pd.DatetimeIndex(data["Datetime"]), inplace=True)
# # VWAP requires the DataFrame index to be a DatetimeIndex.
# # New Columns with results
            data.ta.vwap(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
            data.to_csv('vwapdata.csv')
            columns_to_round = ['ema', 'short_sma', 'long_sma', 'vwap']
            data[columns_to_round] = data[columns_to_round].round(2)

            today=datetime.now().strftime("%Y-%m-%d")
            filtered_data = data[(data['Datetime'] >= today)  ]
            filtered_data=filtered_data.head(1)
            filtered_data.rename(columns={'Open': 'first_open' }, inplace=True)
            filtered_data.rename(columns={'Low': 'first_low' }, inplace=True)
            filtered_data.rename(columns={'Close': 'first_close' }, inplace=True)


            
        
            data = pd.merge(data, filtered_data[[ 'Ticker_Symbol', 'first_open','first_low','first_close']],
                       on=[ 'Ticker_Symbol'], how='left')
            data['first_earning']=data['Close'].astype(float)-data['first_open'].astype(float)
            data['first_earning'] = data['first_earning'].round(2)
            data['start_earning']=data['first_close'].astype(float)-data['first_open'].astype(float)
            data['start_earning'] = data['start_earning'].round(2)
            # data = data[data['first_open'].astype(float) <data['first_low'].astype(float)]


        data["Last_Day_Close"] = (data.groupby("Ticker_Symbol")["Close"].shift(1).astype(float))
        
        for i in range(1, 6):
            column_name = f"Close_{i}d_ago"
            data[column_name] = (
                data.groupby("Ticker_Symbol")["Close"].shift(i).astype(float)
            )
        # data.to_csv("Stocks.csv", index=True)
        for i in range(1, 6):
            column_name = f"Open_{i}d_ago"
            data[column_name] = (
                data.groupby("Ticker_Symbol")["Open"].shift(i).astype(float)
            )
        
        
        # Filter and save latest data for each ticker
        if mode =="master_stock_selection":
            latest_data = data.groupby("Ticker_Symbol").tail(1)
        else:
            interval_minutes=5
            rsi_period=14
            data["rsi"] = calculate_rsi(data, rsi_period)
            latest_data = data.groupby(data['Datetime'].dt.floor(f'{interval_minutes}min')).last()        
            latest_data = latest_data.tail(2).head(1)

        latest_data.loc[:, "Rise_1d_ago"] = (
            latest_data["Close_1d_ago"] > latest_data["Open_1d_ago"]
        ).astype(int)
        latest_data.loc[:, "Rise_2d_ago"] = (
            latest_data["Close_2d_ago"] > latest_data["Open_2d_ago"]
        ).astype(int)
        latest_data.loc[:, "Rise_3d_ago"] = (
            latest_data["Close_3d_ago"] > latest_data["Open_3d_ago"]
        ).astype(int)
        latest_data.loc[:, "Rise_4d_ago"] = (latest_data["Close_4d_ago"] > latest_data["Open_4d_ago"]).astype(int)
        latest_data.loc[:, 'Rise_5d_ago'] = (latest_data['Close_5d_ago'] > latest_data['Open_5d_ago']).astype(int) 

        latest_data.loc[:, "Rise_count"] =  latest_data.loc[
    :, ["Rise_1d_ago", "Rise_2d_ago", "Rise_3d_ago", "Rise_4d_ago", "Rise_5d_ago"]
].sum(axis=1)
        latest_data.loc[:,"Earnings"] = ( latest_data["Close_1d_ago"].astype(float) - latest_data["Open_1d_ago"].astype(float) )
        latest_data["Earnings"] = latest_data["Earnings"].round(2)

        
        columns = [ "Date", "Ticker_Symbol",  "Volume",
    "ema",
    "short_sma",
    "long_sma",
    "vwap",
    "rsi",
    "High",
    "Low",
    "Last_Day_Close",
    "Open",  "Close",
    
    "first_open",
    "start_earning",
    "first_close",
    "first_earning",
    "create_date_time"]
        if mode =='master_stock_selection':
            columns = ["Date",    "Ticker_Symbol",    "Volume",       "High",    "Low",    "Last_Day_Close",    "Open",    "Close",    "Close_1d_ago",    "Close_2d_ago",    "Close_3d_ago",    "Close_4d_ago",    "Close_5d_ago",    "Open_1d_ago",    "Open_2d_ago",    "Open_3d_ago",    "Open_4d_ago",    "Open_5d_ago",    "Rise_1d_ago",    "Rise_2d_ago",    "Rise_3d_ago",    "Rise_4d_ago",    "Rise_5d_ago",  "Rise_count",    "Earnings",    "create_date_time"]

        latest_data = latest_data.reindex(columns=columns)
        # convert the 'Date' column to a string format
        latest_data["Date"] = latest_data["Date"].astype("str")
        
        #logging.info(latest_data.columns)
        if mode != "master_stock_selection":
            latest_data["datetime"]=latest_data["Date"]
            latest_data["volume"]=latest_data["Volume"]

        latest_data["create_date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #latest_data = latest_data.sort_values("Earnings", ascending=False)
        
        latest_data = latest_data[latest_data['Ticker_Symbol'] != ""]

        if mode == "master_stock_selection":
            latest_data = latest_data[latest_data['Rise_count'].astype(float) > 3]
            latest_data = latest_data[latest_data['Earnings'].astype(float) > 1.99]
            latest_data = latest_data[latest_data['Close_1d_ago'].astype(float) > 450]
            latest_data = latest_data[latest_data['Close_1d_ago'].astype(float) <1500]
            latest_data = latest_data[latest_data['Volume'].astype(float) > 3000]
        #else:
         # latest_data = latest_data[latest_data['first_earning'].astype(float) > 1.99]
         # latest_data = latest_data[latest_data['first_open'].astype(float) > 450]
         # latest_data = latest_data[latest_data['first_open'].astype(float) <1100]



        # latest_data = latest_data[latest_data['Open_1d_ago'] > 100]
        # latest_data = latest_data[latest_data['Open_1d_ago'] < 2500]
        # latest_data = latest_data[latest_data['Open'] < latest_data['Close_1d_ago']]

        latest_data = latest_data[latest_data['Ticker_Symbol'].notna()]
        # append_row_to_csv(latest_data)
        
        if mode != "master_stock_selection":
            latest_data.rename(columns={"Last_Day_Close": "last_d_close"}, inplace=True)
            #latest_data.rename(columns={"Volume": "volume"}, inplace=True)
            latest_data.rename(columns={"Earnings": "first_earning"}, inplace=True)

            # latest_data.rename(columns={"Date": "datetime"}, inplace=True)
            latest_data.drop("last_d_close", axis=1, inplace=True)
            latest_data.drop("Volume", axis=1, inplace=True)

            columns=['datetime', 'Ticker_Symbol', 'volume', 'ema', 'short_sma', 'long_sma',
       'vwap','rsi', 'High', 'Low', 'Open', 'Close', 
         "start_earning",
    "first_close",
         'first_open','first_earning', 'create_date_time' ]
            latest_data = latest_data[['datetime', 'Ticker_Symbol', 'volume', 'ema', 'short_sma', 'long_sma',
       'vwap','rsi', 'High', 'Low', 'Open', 'Close', 
         "start_earning",
    "first_close",
         'first_open','first_earning', 'create_date_time']]

            #latest_data = latest_data.reindex(columns=columns)
        finaldata = pd.concat([finaldata, latest_data])
        if not latest_data.empty:
            write_stock_selections(mode, latest_data)

    print("Program ended")
    end_time = time.time()
    total_run_time = end_time - start_time
    print(f"Total program run time: {total_run_time/60.0:.2f} minutes")
    logger.info(mode+" executed successfully.")
    logger.info(f"Total program run time: {total_run_time/60.0:.2f} minutes")

    # return latest_data


def delete_and_create_csv():
    file_path = "StockLists.csv"
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


@app.route("/checkprocess", methods=["GET"])
def check_process():
    global p
    if p is not None and p.is_alive():
        return "Process is running"
    else:
        return "Process has completed"


@app.route("/stock_selection")
# @timeout_decorator.timeout(1800)
def StockSelectionProgram( ):

    
    with app.app_context():
    # Download stock data using yf.
        global p
        try:
            # delete_and_create_csv()
            mode="master_stock_selection"
            executor.submit(get_latest_price(mode))
            return "Task started"
        except Exception as e:
        # Handle the exception
            return jsonify({"error": str(e)})

@app.route("/real")
def RealTimeProgram( ):
    with app.app_context():
    # Download stock data using yf.
        global p
        try:
            # delete_and_create_csv()
            mode="real_time_stock_data"
            executor.submit(get_latest_price(mode))
            return "Task started"
        except Exception as e:
        # Handle the exception
            return jsonify({"error": str(e)})

def get_master_stocks():
    with app.app_context():
        master_stocks = MasterStock.query.all()
        symbol_list = [stock.symbol for stock in master_stocks]
        symbol_list = set(symbol_list)
        symbol_list = list(symbol_list)
    return symbol_list


def get_stocks_selection():
    with app.app_context():
        # master_stocks = MasterStockSelection.query.all()
        master_stocks = MasterStockSelection.query.all()
        symbol_list = [stock.Ticker_Symbol for stock in master_stocks]
        symbol_list = set(symbol_list)
        symbol_list = sorted(symbol_list)
        symbol_list = list(symbol_list)
    return symbol_list

def get_master_stock_data(id):
    # Replace with your actual database connection URL
    database_url = "sqlite:///masterstocks.db"
    engine = create_engine(database_url)

    # Create a session
    Session = sessionmaker(bind=engine)
    dbsession = Session()

    if id != 0:
        master_selections = MasterStockSelection.query.filter_by(id=id).all()
    else:
        master_selections = MasterStockSelection.query.all()

    masters = MasterStock.query.all()

    # Create a dictionary to map Ticker_Symbol to MasterStock records
    symbol_dict = {master.symbol: master for master in masters}

    # Perform the join and populate a list of dictionaries with all columns from MasterStockSelection
    joined_data = []
    for master_selection in master_selections:
        symbol_master = symbol_dict.get(master_selection.Ticker_Symbol)
        if symbol_master:
            # Create a dictionary with all columns from MasterStockSelection
            data = {key: getattr(master_selection, key) for key in master_selection.__table__.columns.keys()}
            
            # Add other columns from MasterStock that you need
            data['company_name'] = symbol_master.company_name
            
            # Add other columns from MasterStock that you need

            joined_data.append(data)

    joined_data = sorted(joined_data, key=lambda x: x['Ticker_Symbol'])
    return joined_data

def get_stock_data(userid):
    database_url = "sqlite:///masterstocks.db"
    engine = create_engine(database_url)


# Create a session
    Session = sessionmaker(bind=engine)
    dbsession = Session()
     # Define the SQL query using SQLAlchemy's text() function, with a parameter for user_id
    query = text('''
        SELECT s.*
        FROM stock_data s,
        (
            SELECT MAX(id) AS MaxID,
                   user_id,
                   symbol
            FROM stock_data
            WHERE user_id = :user_id
            GROUP BY user_id, symbol
        ) t
        WHERE s.id = t.MaxID
    ''').params(user_id=userid)

    # Execute the SQL query and retrieve the results
    result = dbsession.execute(query)
    max_id_data = result.fetchall()

    # Close the session
    dbsession.close()
    #max_id_data = pd.DataFrame(max_id_data)
    logging.info(max_id_data)
    return max_id_data



def get_realtime_stocks(id):
    # Replace with your actual database connection URL
    database_url = "sqlite:///masterstocks.db"
    engine = create_engine(database_url)

# Create a session
    Session = sessionmaker(bind=engine)
    dbsession = Session()
    if(id!=0):
        stocks= RealTimeStockData.query.filter_by(id=id).all()
    else:
        stocks = RealTimeStockData.query.all()
    
    
    masterstockselection  = MasterStockSelection.query.all()

   
    masters=MasterStock.query.all()
    

# Create dictionaries to map Ticker_Symbol to MasterStockSelection records and symbol to MasterStock records
    master_selection_dict = {master.Ticker_Symbol: master for master in masterstockselection}
    symbol_dict = {master.symbol: master for master in masters}
    



    # Perform the join and populate a list of dictionaries with all columns
    joined_data = []
    for stock in stocks:
            master_selection = master_selection_dict.get(stock.ticker_symbol)
            symbol_master = symbol_dict.get(stock.ticker_symbol)
            if master_selection and symbol_master:
                close_average=(master_selection.Close_1d_ago + master_selection.Close_2d_ago + master_selection.Close_3d_ago +
                     master_selection.Close_4d_ago + master_selection.Close_5d_ago) / 5
                
                average_open=(master_selection.Open_1d_ago + master_selection.Open_2d_ago + master_selection.Open_3d_ago +
                     master_selection.Open_4d_ago + master_selection.Open_5d_ago) / 5

                close_average = round(close_average, 2)
                average_open = round(average_open, 2)

                earning_current=(stock.close-stock.open)
                earning_current = round(earning_current, 2)
                stock_rsi= stock.rsi 
                signal=''  


                joined_data.append({
              **master_selection.__dict__,
             'Rise_count': master_selection.Rise_count ,
            **stock.__dict__,
              'close_average': close_average,
              'average_open':average_open,
              'company_name': symbol_master.company_name,
             'icici_symbol': symbol_master.icici_symbol,
             'current_earning':earning_current,
             'stock_rsi':stock_rsi,
             'signal':signal
        })

    sorted_data = sorted(joined_data, key=lambda x: x['start_earning'],reverse=True)
 
    return sorted_data

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/jobstatus")
def jobs():
    if 'email' not in session:
        return redirect(url_for("login"))
    # Get the user's email from the session
    email = session.get( "email")
    # Find the user with the provided email
    user = Users.query.filter_by(email=email).first()
    job_list = []
    jobs = scheduler.get_jobs()
    for job in jobs:
        job_dict = {}
        job_dict["id"] = job.id
        job_dict["name"] = job.name
        job_dict["next_run_time"] = job.next_run_time
        if job.next_run_time is not None:
            job_dict["status"] = "Running"
        else:
            job_dict["status"] = "Not running"
        job_list.append(job_dict)
    return render_template("jobs.html", jobs=job_list, user=user)

def cursor_to_list(cursor):
    return [doc for doc in cursor]

# Function to determine the Remark based on conditions
def determine_remark(row):
    
    if (row["first_open"] is not None and 
        row["Close"] is not None and
        row["first_open"] > row["Close"] ):
        return "PRE SELL"
    elif (
        row["Close"] is not None and
        row["ema"] is not None and
        row["vwap"] is not None and
        row["short_sma"] is not None and
        row["long_sma"] is not None and
        row["Close"] > row["ema"] and
        row["vwap"] > 0 and
        row["short_sma"] > row["long_sma"]
    ):
        return "BUY"
     
    
    else:
        return "WAIT"
    
@app.route("/tradingwindow")
def tradingwindow():
    if 'email' in session:
        email = session.get( "email")
        user = Users.query.filter_by(email=email).first()
        if user:
         user_id = user.id
         print(f"The user's ID is: {user_id}")
    else:
        return redirect(url_for("login"))

    headers = [ "Id","Date and Time","Symbol","Name Of Company","Volume","Ema","Short EMA","Long Ema", "Morning Open Value","Current Open Value", "Current Close Value", "QUANTUM", "Average Value", "Total Value", "Remark", "Signal"]
 
    # # Connect to the SQLite database using sqlite3
    try:
         
        joined_data=get_realtime_stocks(0)
         
        for row in joined_data:
            row["signal"] = determine_remark(row)
        
        # stock_data=get_stock_data(user_id)
        
        # if len(stock_data) >0:
            # df_joined = pd.DataFrame(joined_data)
            # df_stock = pd.DataFrame(stock_data)
            # df_stock= df_stock[['symbol', 'user_id', 'signal', 'average_value']]
 
            # left_join_columns = ['ticker_symbol',   'signal']  # Columns from df_joined
            # right_join_columns = ['symbol', 'signal']  # Columns from df_stock

        
            # result_df = pd.merge(df_joined, df_stock, left_on=left_join_columns, right_on=right_join_columns, how='left')
       
            # result_df['average_value'] =  result_df['average_value'].fillna('0')
            # joined_data = result_df.to_dict('records')

        for row in joined_data:
            row["signal"] = determine_remark(row)
        total=len(joined_data)
        if joined_data:
        # if not joined_data.empty:
            # StartLoadedDate = min(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
            # EndLoadedDate = max(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
            StartLoadedDate=None
            EndLoadedDate=None
            # StartLoadedDate=joined_data['create_date_time'].min()
            # EndLoadedDate=joined_data['create_date_time'].max()

        else:
            StartLoadedDate=None
            EndLoadedDate=None
        # max_create_date_time = datetime.strptime(EndLoadedDate, '%Y-%m-%d %H:%M:%S')
        # min_create_date_time = datetime.strptime(StartLoadedDate, '%Y-%m-%d %H:%M:%S')
        # time_difference = max_create_date_time - min_create_date_time
        # TimeTakeninMin = time_difference.total_seconds() / 60
        TimeTakeninMin=None
         
       

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # # Close the connection
        logging.info("Time to CLose Db")
        # conn.close()
    print(joined_data)
    
    return render_template(
        "trading_window.html",
        stocks=joined_data,
        user=user,
        total=total,
        StartLoadedDate=StartLoadedDate,
        EndLoadedDate=EndLoadedDate,
        TimeTakeninMin=TimeTakeninMin,
        headers=headers 
    )


@app.route("/realtimestocks")
def realtimestocks():
    if 'email' in session:
        email = session.get( "email")
        user = Users.query.filter_by(email=email).first()
    else:
        return redirect(url_for("login"))

    try:
        joined_data=get_realtime_stocks(0)
        if joined_data:
            total=len(joined_data)
            StartLoadedDate = min(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
            EndLoadedDate = max(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
            max_create_date_time = datetime.strptime(EndLoadedDate, '%Y-%m-%d %H:%M:%S')
            min_create_date_time = datetime.strptime(StartLoadedDate, '%Y-%m-%d %H:%M:%S')
            time_difference = max_create_date_time - min_create_date_time
            TimeTakeninMin = time_difference.total_seconds() / 60
        else:
            total=0
            StartLoadedDate = "NA"
            EndLoadedDate = "NA"
           
            TimeTakeninMin = "NA"


    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # # Close the connection
        logging.info("Time to CLose Db")
        # conn.close()
  
    return render_template(
        "realtime_stocks.html",
        stocks=joined_data,
        user=user,
        total=total,
        StartLoadedDate=StartLoadedDate,
        EndLoadedDate=EndLoadedDate,
        TimeTakeninMin=TimeTakeninMin
    )


@app.route("/stocks")
def stocks():
    if 'email' in session:
        email = session.get( "email")
        user = Users.query.filter_by(email=email).first()
    else:
        return redirect(url_for("login"))

    # # Connect to the SQLite database using sqlite3
    try:
        joined_data=get_master_stock_data(0)
        total=len(joined_data)
        StartLoadedDate = min(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
        EndLoadedDate = max(joined_data, key=lambda x: x['create_date_time'])['create_date_time']
        max_create_date_time = datetime.strptime(EndLoadedDate, '%Y-%m-%d %H:%M:%S')
        min_create_date_time = datetime.strptime(StartLoadedDate, '%Y-%m-%d %H:%M:%S')
        time_difference = max_create_date_time - min_create_date_time
        TimeTakeninMin = time_difference.total_seconds() / 60

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # # Close the connection
        logging.info("Time to CLose Db")
        # conn.close()
    return render_template(
        "stocks.html", stocks=joined_data, user=user, total=total, LoadedDate=EndLoadedDate
    )


@app.route("/dashboard")
def dashboard():
    if session:
        print(session)
    email = session.get( "email" )
     

    if email:
        user = Users.query.filter_by(email=email).first()
        return render_template("dashboard.html", user=user)
    else:
        return redirect(url_for("login"))

    

@app.route("/list")
@app.route("/list/<int:page>")
def home(page=1):
    if "email" not in session:
        return redirect(url_for("login"))
    email = session.get( "email")

    user = Users.query.filter_by(email=email).first()
    stocks = MasterStock.query.paginate(page=page, per_page=10)
    
    return render_template("list.html", stocks=stocks, user=user, pagination=stocks)


@app.route("/admin/users")
def user_list():


    # Get all the users from the database
    users = Users.query.all()
    # Render the user list template with the list of users
    if "email" not in session:
        return redirect(url_for("login"))
    # Get the user's email from the session
    email = session.get( "email")

    # Find the user with the provided email
    user = Users.query.filter_by(email=email).first()
    print(users)
    return render_template("user_list.html", users=users, user=user)


@app.route("/admin/users/<int:user_id>/edit", methods=["GET", "POST"])
def edit_user(user_id):
    user = Users.query.get(user_id)
    if request.method == "POST":
        user.email = request.form["email"]
        user.password = request.form["password"]
        user.mobile_no = request.form["mobile_no"]
        user.address = request.form["address"]

        user.is_admin = "is_admin" in request.form
        db.session.commit()
        return redirect(url_for("dashboard"))
    else:
        return render_template("edit_user.html", user=user)


@app.route("/search")
def search():
    query = request.args.get("search")
    # Check if the user is logged in
    if "email" not in session:
        return redirect(url_for("login"))
    # Get the user's email from the session
    email = session.get("email")
    # Find the user with the provided email
    user = Users.query.filter_by(email=email).first()

    filtered_stocks = MasterStock.query.filter(
        MasterStock.symbol.ilike(f"%{query}%")
        | MasterStock.company_name.ilike(f"%{query}%")
    ).all()
    return render_template("list.html", stocks=filtered_stocks, user=user)


@app.route("/add", methods=["GET", "POST"])
def add():
    # Check if the user is logged in
    if "email" not in session:
        return redirect(url_for("login"))
    # Get the user's email from the session
    email = session.get("email")
    # Find the user with the provided email
    user = Users.query.filter_by(email=email).first()
    if request.method == "POST":
        symbol = request.form["symbol"]
        company_name = request.form["company_name"]
        icici_symbol = request.form["icici_symbol"]
        stock = MasterStock(
            symbol=symbol, company_name=company_name, icici_symbol=icici_symbol
        )
        db.session.add(stock)
        db.session.commit()
        return redirect(url_for("home"))
    else:
        return render_template("add.html", user=user)



@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):
    stock = MasterStock.query.get(id)
    if request.method == "POST":
        stock.symbol = request.form["symbol"]
        stock.company_name = request.form["company_name"]
        stock.icici_symbol = request.form["icici_symbol"]
        db.session.commit()
        return redirect(url_for("home"))
    else:
        return render_template("edit.html", stock=stock)


@app.route("/delete/<int:id>", methods=["GET", "POST"])
def delete(id):
    stock = MasterStock.query.get(id)
    db.session.delete(stock)
    db.session.commit()
    return redirect(url_for("home"))

# In your Flask app
# ...

def find_user_by_email(email):
    with app.app_context():
        user = Users.query.filter_by(email=email).first()
    return user

def convert_to_string(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')   

@app.route('/stockdata')
def stockdatabuysale():
    url = "sqlite:///" + os.path.join(basedir, "masterstocks.db")
    print(url)
    engine = create_engine(url)
    conn = engine.connect()
    query = text("SELECT * FROM stock_data   ")
    stocks = conn.execute(query)
    stocks_data = []

# Convert the query result to a list of dictionaries
    for row in stocks:
        row_dict = dict(row)
        stocks_data.append(row_dict)
    # Convert the list of dictionaries to JSON
    stocks_data_json = json.dumps(stocks_data)

    print(stocks_data_json)
    return stocks_data_json

def recalculate_average_values():
    # Define a custom SQL query with a window function to calculate the average_value
    query = text('''
        UPDATE stock_data
        SET average_value = (
    SELECT AVG(total_value)
    FROM stock_data AS subquery
    WHERE symbol = stock_data.symbol 
      AND signal = stock_data.signal 
      AND user_id = stock_data.user_id 
      AND date_time <= stock_data.date_time
);

    ''')

    # Execute the custom SQL query
    db.session.execute(query)
    db.session.commit()   

@app.route('/action/<int:id>', methods=['GET'])
def tradingwindowstock(id):
    print(id)
    user=''
    if 'email' in session:
        # Retrieve the email ID from the session
        email = session['email']
        user=session["user"]
    else:
        return redirect(url_for("login"))

    try:
        stock=get_realtime_stocks(id)
         


    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        # # Close the connection
        logging.info("Time to CLose Db")
        # conn.close()

    stock=  [item for item in stock if item['id'] == id] 

    for row in stock:
        row["signal"] = determine_remark(row)
        row["total_value"] = 1.0 * float(row["close"]) 
        
        if row["signal"] == "BUY":
            row["buy_amt"] = row["total_value"]
            row["sell_amt"] = 0
        elif row["signal"] == "SELL":
            row["sell_amt"] = row["total_value"]
            row["buy_amt"] = 0
        elif row["signal"] == "PRE SELL":
            row["sell_amt"] = row["total_value"]
            row["buy_amt"] = 0
        else:
            row["buy_amt"] = 0
            row["sell_amt"] = 0
 

    stock=stock[0]
    if stock:
        # Select specific columns using with_entities
        selected_columns = [
            stock['datetime'],
            stock['ticker_symbol'],
            stock['company_name'],
            stock['close'],
            stock['signal'],
            stock['total_value'],
            stock['buy_amt'],
            stock['sell_amt']

        ]
 
 
    print(user)
    datetime_string = selected_columns[0] 
    stock_data_object = StockData(
            user_id=user['id'],
            date_time=datetime_string,
            symbol=selected_columns[1],
            name_of_company=selected_columns[2],
            rate=selected_columns[3],
            quantum=1,
            total_value=selected_columns[5],
            remark="",

            signal=selected_columns[4],
            average_value=0,
            buy_amt=selected_columns[6] ,
            sell_amt=selected_columns[7] ,
            balance=0
        )
     
    db.session.add(stock_data_object)
    db.session.commit()
    recalculate_average_values()

    # Return a JSON response indicating the success of the action
    return redirect('/dashboard')
    # return jsonify({'success': True, 'message': 'Action completed successfully'})


@app.route("/upload-csv", methods=["GET", "POST"])
def upload_csv():
    # Check if the user is logged in
    if "email" not in session:
        return redirect(url_for("login"))
    # Get the user's email from the session
    email = session.get("email")
    # Find the user with the provided email
    user = Users.query.filter_by(email=email).first()
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template(
                "upload_csv.html", error="No file selected", user=user
            )
        if file:
            filename = file.filename
            file.save(filename)
            with open(filename, "r") as csvfile:
                csvreader = csv.reader(csvfile)
                next(csvreader)  # skip header row
                for row in csvreader:
                    stock = MasterStock.query.filter_by(symbol=row[0]).first()
                    if stock:
                        stock.icici_symbol = row[1]
                        stock.company_name = row[2]
                    else:
                        stock = MasterStock(
                            symbol=row[0], company_name=row[1], icici_symbol=row[2]
                        )
                        db.session.add(stock)
                db.session.commit()
            return render_template(
                "upload_csv.html", success="CSV file uploaded successfully", user=user
            )
    return render_template("upload_csv.html", user=user)

# New route and view function to handle CSV export request
@app.route('/export_csv', methods=['POST'])
def export_csv():
    data = request.json['data']

    # Create a CSV file in-memory
    csv_data = '\n'.join(data)

    # Prepare the response with the CSV data
    # Save the DataFrame to a CSV file on the server
    csv_filename = 'stock_data.csv'
    csv_path = os.path.join(app.root_path, csv_filename)
    csv_data.to_csv(csv_path, index=False)
    response = Response(
        csv_data,
        headers={
            "Content-Disposition": "attachment; filename=stock_data.csv",
            "Content-type": "text/csv",
        }
    )

    return response

@app.route('/download_csv')
def download_csv():
    # Function to generate CSV data from the sample_data
    def generate_csv():
        # Create a file-like object to store the CSV data
        csv_output = io.StringIO()

        # Create a CSV writer
        csv_writer = csv.writer(csv_output, delimiter=',')

        # Write the CSV header
        csv_writer.writerow(["Id", "Symbol", "ICICI Symbol", "Company Name"])
        
        master_stocks = MasterStock.query.all()

# Convert the list of "MasterStock" objects into a list of dictionaries
        master_stocks_list = [
                    {
                        "id": stock.id,
                        "symbol": stock.symbol,
                        "icici_symbol": stock.icici_symbol,
                        "company_name": stock.company_name,
                    }
            for stock in master_stocks
            ]

        for stock in master_stocks_list:
            csv_writer.writerow([stock["id"], stock["symbol"], stock["icici_symbol"], stock["company_name"]])

        # Get the CSV data as a string
        csv_data = csv_output.getvalue()

        return csv_data

    # Create a response with the CSV data and appropriate headers
    response = Response(generate_csv(), content_type='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=master_stocks.csv'

    return response

@app.route('/account_statment')
def account_statment():
    email = session.get("email")
    user = Users.query.filter_by(email=email).first()
    stock_data = StockData.query.all()
    return render_template('account_statement.html', stock_data=stock_data,user=user)
    
# Working
# scheduler.add_job(func=run_script2,trigger='date', run_date=datetime(2023, 7, 14, 7, 30, 0))

# Runs from Monday to Friday at 5:30 (am) until 2014-05-30 00:00:00
# scheduler.add_job( StockSelectionProgram  , "cron", day_of_week="mon-sun", hour=16,minute=33,end_date="2023-12-30")
scheduler.add_job( StockSelectionProgram  , "cron", day_of_week="mon-fri", hour=8,minute=15,end_date="2023-12-30")
# scheduler.add_job( RealTimeProgram  , "cron", day_of_week="mon-sun", hour=16,minute=45,end_date="2023-12-30")
scheduler.add_job(RealTimeProgram, "cron", day_of_week="mon-fri", hour="9-15", minute="18/5")
scheduler.start() 

if __name__ == "__main__":
    p = None
    app.run(debug=false)

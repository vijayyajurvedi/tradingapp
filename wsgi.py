from flaskapp import app
if __name__ == "__main__":
    context=('/home/gouriimp/public_html/rudra-t-s/project/certs/domain.com.crt','/home/gouriimp/public_html/rudra-t-s/project/certs/domain.com.key')
    app.run(host="ras2v-tr.gouriimpex.com",port=8082,debug=True, ssl_context=context) 
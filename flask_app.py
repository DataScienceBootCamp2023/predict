###############################################################################
#                            RUN MAIN                                         #
###############################################################################

# setup
## pkg
import flask
import pandas as pd

from model.credit_card_default_prediction import CreditCardDefaultPrediction
from settings import config

## app
app = flask.Flask(__name__, 
				instance_relative_config=True, 
       			template_folder=config.root+'client/templates',
                static_folder=config.root+'client/static')



# main
@app.route("/", methods=['GET','POST'])
def index():
	try:
		if flask.request.method == 'POST':
			## data from client
			## data from client
			app.logger.info(flask.request.files)
			app.logger.info(flask.request.form)
			data = pd.read_excel(flask.request.files["data"])
			model_name = flask.request.form["model"]
			app.logger.warning("--- Inputs Received ---")
					## predict
			model = CreditCardDefaultPrediction(data)
			predictions = model.predict(model_name)
			xlsx_out = model.write_excel(predictions)
			return flask.send_file(xlsx_out, attachment_filename='CreditCardDefaultPrediction.xlsx', as_attachment=True)             
		else:
			return flask.render_template("index.html")

	except Exception as e:
		app.logger.error(e)
		flask.abort(500)
 
# errors
@app.errorhandler(404)
def page_not_found(e):
    return flask.render_template("errors.html", msg="Page doesn't exist"), 404
    

@app.errorhandler(500)
def internal_server_error(e):
    return flask.render_template('errors.html', msg="Something went terribly wrong"), 500


# run
if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.debug)
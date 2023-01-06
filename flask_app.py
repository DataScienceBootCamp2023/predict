###############################################################################
#                            RUN MAIN                                         #
###############################################################################

# setup
## pkg

import flask
import pandas as pd
from sklearn.externals import joblib
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
    if flask.request.method == 'POST':
        # Get the input file and model file from the request
        input_file = flask.request.files['input_file']
        model_file = flask.request.files['model_file']
        
        # Load the model from the model file
        model = CreditCardDefaultPrediction(model_file)
        
        # Process the input data
        input_data = pd.read_excel(input_file)
        
        # Make predictions using the model
        output_data = model.predict(input_data)
        
        # Create an in-memory buffer to store the output data
        output_buffer = io.BytesIO()
        
        # Write the output data to the buffer
        output_df = pd.DataFrame(output_data)
        output_df.to_excel(output_buffer, index=False)
        
        # Rewind the buffer to the beginning
        output_buffer.seek(0)
        
	# Send the output data as a file download
	return flask.send_file(output_buffer, attachment_filename='predictions.xlsx', as_attachment=True)
	    else:
		return flask.render_template('index.html')

	if __name__ == '__main__':
	    app.run(host=config.host, port=config.port, debug=config.debug)

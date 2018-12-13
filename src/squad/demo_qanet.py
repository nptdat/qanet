#############
# Run:
# $ FLASK_APP=demo_qanet.py flask run --host=0.0.0.0
#############

from flask import Flask
from flask import render_template
from flask import request

from inference import load_model

app = Flask(__name__)


model = load_model()

@app.route('/qanet', methods=['GET', 'POST'])
def qanet():
    print('METHOD:', request.method)
    if request.method == 'GET':
        # Default inputs
        # context = "In early 2012, NFL Commissioner Roger Goodell stated that the league planned to make the 50th Super Bowl \"spectacular\" and that it would be \"an important game for us as a league\"."
        # query = "Which Super Bowl did Roger Goodell speak about?"
        # answer = ' '.join(model.ask(context, query))

        context = ''
        query = ''
        answer = ''
    else:
        print('--- Process as POST')
        context = request.form['context']
        query = request.form['query']
        answer = ' '.join(model.ask(context, query))
        print('Context:', context)
        print('Query:', query)
        print('Answer:', answer)

    return render_template(
        'qanet.html',
        context=context,
        query=query,
        answer=answer
    )

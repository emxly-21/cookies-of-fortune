from flask import Flask
from flask_ask import Ask, question, statement
import generator

app = Flask(__name__)
ask = Ask( app,"/")

@ask.launch
def launch_app():
    return question("Would you like to hear your fortune?")

@ask.intent("FortuneIntent")
def give_fortune():
    return statement(generator.tell_fortune())

if __name__ == "__main__":
    app.run(debug=True)
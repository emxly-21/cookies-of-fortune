from flask import Flask
from flask_ask import Ask, question, statement
import generator
import random

app = Flask(__name__)
ask = Ask(app, "/")

@ask.launch
def launch_app():
    return question("Would you like to hear your fortune?")

@ask.intent("FortuneIntent")
def give_fortune():
    return statement(generator.tell_fortune())

@ask.intent("CowardIntent")
def coward():
    choice = random.randint(0,2)
    if choice == 0:
        phrase = "You coward!"
    elif choice == 1:
        phrase = "I pity the fool!"
    else:
        phrase = "bye Felisha"
    return statement(phrase)
"""
@ask.intent("CancelIntent")
def bye():
    return statement("Goodbye")
"""
if __name__ == "__main__":
    app.run(debug=True)
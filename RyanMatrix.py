import Modes as modes
import board
from flask import Flask, request
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
load_dotenv()
from os import getenv


LED_PIN = board.D18
NUM_LEDS = 512
BRIGHTNESS_PERCENT = 10
NUM_ROWS = 16
NUM_COLS = 32
WEATHER_API_KEY = getenv("WEATHER_API_KEY")
WEATHER_CITY = "Plano"
WEATHER_STATE = "TX"
WEATHER_COUNTRY = "US"
WEATHER_UNITS = "imperial"


modes.init(LED_PIN, NUM_LEDS, BRIGHTNESS_PERCENT, NUM_ROWS, NUM_COLS, False)
last_mode = None


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/getmodes", methods = ["GET"])
@cross_origin()
def send_modes_object():
    return """
    {
        "off": {},
        "time": { "color": true },
        "temperature": {},
        "pride": {},
        "fill and unfill": { "color": true },
        "sparkle": { "color": true },
        "theater chase rainbow": {},
        "rainbow cycle": {},
        "camera": {},
        "text": { "color": true, "textOrURL": true },
        "rainbow text": { "textOrURL": true },
        "scrolling text": { "color": true, "textOrURL": true },
        "load image url": { "textOrURL": true },
        "load video url": { "textOrURL": true },
        "equalizer": { "color": true },
        "rainbow equalizer": {},
        "object detection": { "color": true }
    }
    """


@app.route("/", methods = ["POST"])
@cross_origin()
def display_mode():
    json = request.json

    mode = json["mode"]
    color = json["color"]
    text = json["text"]

    global last_mode
    if mode == last_mode:
        modes.current_function = modes.blank
        modes.current_function()

    last_mode = mode
    color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    if mode == "off":
        modes.current_function = modes.blank
        modes.current_function()
    if mode == "time":
        modes.current_function = modes.clock
        modes.current_function(color)
    if mode == "temperature":
        modes.current_function = modes.temperature
        modes.current_function(WEATHER_CITY, WEATHER_STATE, WEATHER_COUNTRY, WEATHER_API_KEY, WEATHER_UNITS)
    elif mode == "pride":
        modes.current_function = modes.load_image_url
        modes.current_function("https://media.them.us/photos/5b1ef721bfa1890010f0e15e/3:2/w_1079,h_719,c_limit/new-pride-flag-01.jpg")
    elif mode == "gang shit":
        modes.current_function = modes.scrolling_text
        modes.current_function("GANG SHIT", color)
    elif mode == "fill and unfill":
        modes.current_function = modes.dual_stripes
        modes.current_function(color, (0, 0, 0))
    elif mode == "sparkle":
        modes.current_function = modes.sparkle
        modes.current_function(color)
    elif mode == "theater chase rainbow":
        modes.current_function = modes.theater_chase_rainbow
        modes.current_function()
    elif mode == "rainbow cycle":
        modes.current_function = modes.rainbow_cycle
        modes.current_function()
    elif mode == "camera":
        modes.current_function = modes.load_camera
        modes.current_function()
    elif mode == "text":
        modes.current_function = modes.text
        modes.current_function(text, color)
    elif mode == "rainbow text":
        modes.current_function = modes.rainbow_text
        modes.current_function(text)
    elif mode == "scrolling text":
        modes.current_function = modes.scrolling_text
        modes.current_function(text, color)
    elif mode == "load image url":
        modes.current_function = modes.load_image_url
        modes.current_function(text)
    elif mode == "load video url":
        modes.current_function = modes.load_video_url
        modes.current_function(text)
    elif mode == "equalizer":
        modes.current_function = modes.equalizer
        modes.current_function(color)
    elif mode == "rainbow equalizer":
        modes.current_function = modes.rainbow_equalizer
        modes.current_function()
    elif mode == "object detection":
        modes.current_function = modes.object_detection
        modes.current_function(color)

    return ""


if __name__ == "__main__":
    app.run()
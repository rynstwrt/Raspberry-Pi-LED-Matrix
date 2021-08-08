import Modes as modes
import board

LED_PIN = board.D18
NUM_LEDS = 512
BRIGHTNESS_PERCENT = 20
NUM_ROWS = 16
NUM_COLS = 32
WEATHER_API_KEY = None
WEATHER_CITY = "Plano"
WEATHER_STATE = "TX"
WEATHER_COUNTRY = "US"
WEATHER_UNITS = "imperial"

COLOR_DEFAULT = (52, 235, 82)
COLOR_BLACK = (0, 0, 0)

modes.init(LED_PIN, NUM_LEDS, BRIGHTNESS_PERCENT, NUM_ROWS, NUM_COLS, False)

try:
    #modes.scrolling_text("Hello, world!", (255, 255, 255))
    modes.temperature(WEATHER_CITY, WEATHER_STATE, WEATHER_COUNTRY, WEATHER_API_KEY, WEATHER_UNITS)
except KeyboardInterrupt:
    modes.blank()

################### IMPORTS ###################
import random
import neopixel
from time import sleep, localtime, strftime
import cv2
from PIL import Image, ImageFont, ImageDraw
import requests
from math import ceil



################### VARIABLES ###################
pixels = None
NUM_LEDS = None
COLOR_BLACK = (0, 0, 0)
NUM_ROWS = None
NUM_COLS = None



################### HELPER FUNCTIONS ###################
def init(led_pin, num_leds, brightness, num_rows, num_cols, auto_write = True):
    global pixels
    global NUM_LEDS
    global NUM_ROWS
    global NUM_COLS

    pixels = neopixel.NeoPixel(led_pin, num_leds, brightness = brightness / 100, auto_write = auto_write)
    NUM_LEDS = num_leds
    NUM_ROWS = num_rows
    NUM_COLS = num_cols


def blank():
    pixels.fill(COLOR_BLACK)
    pixels.show()


def get_row(index):
    return index // NUM_ROWS


def get_col(index):
    row = get_row(index)
    return index - row * NUM_ROWS if row % 2 == 0 else NUM_ROWS - (index - row * NUM_ROWS)


def wheel(pos):
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos * 3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos * 3)
        g = 0
        b = int(pos * 3)
    else:
        pos -= 170
        r = 0
        g = int(pos * 3)
        b = int(255 - pos * 3)
    return (r, g, b)



################### MODE FUNCTIONS ###################
def dual_stripes(color1, color2):
    while True:
        for i in range(NUM_LEDS):
            pixels[i] = color1
            pixels.show()
        for i in range(NUM_LEDS):
            pixels[i] = color2
            pixels.show()


def dual_fill(color1, color2):
    while True:
        pixels.fill(color1)
        pixels.show()
        sleep(.5)
        pixels.fill(color2)
        pixels.show()
        sleep(.5)


def rainbow_cycle():
    for j in range(255):
        for i in range(NUM_LEDS):
            pixel_index = (i * 256 // NUM_LEDS) + j
            pixels[i] = wheel(pixel_index & 255)
            pixels.show()
        sleep(2)


def theater_chase_rainbow(wait_ms=50):
    for j in range(256):
        for q in range(3):
            for i in range(0, NUM_LEDS, 3):
                pixels[i+q - 1] = wheel((i+j) % 255)
            pixels.show()
            sleep(wait_ms/1000.0)
            for i in range(0, NUM_LEDS, 3):
                pixels[i+q - 1] = (0, 0, 0)
    theater_chase_rainbow(wait_ms)


def load_image_url(url):
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)

    for i in range(NUM_LEDS):
        col = NUM_COLS - get_row(i) - 1
        row = NUM_ROWS - get_col(i) - 1

        coord = col, row
        pixels[i] = img.getpixel(coord)
    
    pixels.show()


def load_image_camera(img_array):
    img = Image.fromarray(img_array).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))


    for i in range(NUM_LEDS):
        col = get_row(i)
        row = NUM_ROWS - get_col(i) - 1

        coord = col, row
        pixels[i] = img.getpixel(coord)
    
    pixels.show()


def load_camera():
    video_capture = cv2.VideoCapture(-1, cv2.CAP_V4L)
    success, img = video_capture.read()

    while success:
        success, img = video_capture.read()

        if not success:
            break

        load_image_camera(img)
    
    video_capture.release()


def text(text, color):
    img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

    text_size = draw.textsize(text, font=font)
    width = text_size[0]
    height = text_size[1]

    draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), text, color, font = font)

    for i in range(NUM_LEDS):
        col = NUM_COLS - get_row(i) - 1
        row = NUM_ROWS - get_col(i) - 1

        coord = col, row
        pixels[i] = img.getpixel(coord)
    
    pixels.show()

    sleep(99999)


def rainbow_text(text):
    index = 0

    img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 14)

    text_size = draw.textsize(text, font=font)
    width = text_size[0]
    height = text_size[1]

    draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), text, (255, 255, 255), font = font)

    while True:
        for i in range(NUM_LEDS):
            col = NUM_COLS - get_row(i) - 1
            row = NUM_ROWS - get_col(i) - 1

            coord = col, row
            pixels[i] = wheel(index + i & 255) if img.getpixel(coord) != (0, 0, 0) else (0, 0, 0)
        
        pixels.show()
        index += 1

        if index > 255:
            index = 0


def scrolling_text(text, color):
    index = -NUM_COLS

    while True:
        img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 14)
        #font = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf", 14)

        text_size = draw.textsize(text, font=font)
        width = text_size[0]
        height = text_size[1]

        draw.text((-index, ceil(NUM_ROWS / 2 - height / 2) - 1), text, color, font = font)

        for i in range(NUM_LEDS):
            col = NUM_COLS - get_row(i) - 1
            row = NUM_ROWS - get_col(i) - 1

            coord = col, row
            pixels[i] = img.getpixel(coord)
        
        pixels.show()
        index += 1

        if index > width:
            index = -NUM_COLS
        sleep(.02)


def pride_flag():
    load_image_url("https://media.them.us/photos/5b1ef721bfa1890010f0e15e/3:2/w_1079,h_719,c_limit/new-pride-flag-01.jpg")


def rainbow_fill():
    index = 0
    while True:
        pixels.fill(wheel(index & 255))
        pixels.show()
        index += 1

        if index > 255:
            index = 0


def display_time(time_string, color):
    img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

    text_size = draw.textsize(time_string, font=font)
    width = text_size[0]
    height = text_size[1]

    draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), time_string, color, font = font)

    for i in range(NUM_LEDS):
        col = NUM_COLS - get_row(i) - 1
        row = NUM_ROWS - get_col(i) - 1

        coord = col, row
        pixels[i] = img.getpixel(coord)
    
    pixels.show()

    sleep(.02)


def clock(color):
    local_time = localtime()
    current_time = strftime("%H:%M", local_time)
    seconds = strftime("%S", local_time)

    display_time(current_time, color)
    sleep(60 - int(seconds))
    display_time(strftime("%H:%M", local_time), color)

    while sleep(60):
        display_time(strftime("%H:%M", local_time), color)


def random_images():
    while True:
        url = "https://picsum.photos/seed/{0}/32/16".format(random.randint(0, 999999))
        load_image_url(url)
        print("Now displaying {0}".format(url))
        input("Press enter to change the image...")


def sparkle(color1):
    while True:
        for i in range(NUM_LEDS):
            pixels[i] = color1 if get_col(i) % 2 == 0 else COLOR_BLACK
        pixels.show()
        sleep(.2)
        for i in range(NUM_LEDS):
            pixels[i] = color1 if get_col(i) % 2 != 0 else COLOR_BLACK
        pixels.show()
        sleep(.2)


def temperature(city, state, country, api_key, units):
    response = requests.get("http://api.openweathermap.org/data/2.5/weather?q={},{},{}&appid={}&units={}".format(city, state, country, api_key, units))
    if (response.status_code != 200):
        pixels.fill((255, 0, 0))
        pixels.show()
        print(response.json())
        return

    temperature = response.json()["main"]["temp"]
    temperature_text = "{:.1f}°".format(temperature)

    color = None
    if temperature > 90:
        color = (255, 0, 0)
    elif temperature < 55:
        color = (31, 154, 255)
    else:
        color = (255, 255, 255)

    text(temperature_text, color)

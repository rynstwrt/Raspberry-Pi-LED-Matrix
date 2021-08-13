import neopixel
import board
from asyncio import sleep, create_task
from PIL import Image, ImageFont, ImageDraw
import requests
import cv2
import pyaudio
import numpy as np
np.set_printoptions(suppress=True)
from scipy import fft



################### VARIABLES ###################
LED_PIN = board.D18
NUM_LEDS = 512
BRIGHTNESS_PERCENT = 10
NUM_ROWS = 16
NUM_COLS = 32

AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 2
AUDIO_RATE = 48000
AUDIO_DEVICE_INDEX = 0
AUDIO_BLOCK_SIZE = 1024 * 2


pixels = neopixel.NeoPixel(LED_PIN, NUM_LEDS, brightness = BRIGHTNESS_PERCENT / 100, auto_write = False)


MODES = """{
            "off": {},
            "rgb": {},
            "theater chase rainbow": {},
            "fill and unfill": { "color": true },
            "sparkle": { "color": true },
            "rainbow cycle": {},
            "scrolling text": { "color": true, "textOrURL": true },
            "rainbow scrolling text": { "textOrURL": true },
            "load image url": { "textOrURL": true },
            "progress pride flag": {},
            "camera": {},
            "equalizer": { "color": true },
            "rainbow equalizer": {}
        }"""

################### HELPER FUNCTIONS ###################
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


def get_pixel_x(index):
    return NUM_COLS - (index // NUM_ROWS) - 1


def get_pixel_y(index):
    return NUM_ROWS - (index % NUM_ROWS) - 1


def get_pixel_index_at(x, y):
    a = (NUM_COLS - 1 - x) * NUM_ROWS
    b = NUM_ROWS - y - 1 if x % 2 != 0 else y
    return (a + b)


async def display_image(image):
    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            index = get_pixel_index_at(x, y)
            coords = x, y
            pixels[index] = image.getpixel(coords)

    pixels.show()


################### MODE FUNCTIONS ###################
async def off():
    pixels.fill((0, 0, 0))
    pixels.show()


async def rgb():
    while True:
        pixels.fill((255, 0, 0 ))
        pixels.show()
        await sleep(1)

        pixels.fill((0, 255, 0))
        pixels.show()
        await sleep(1)

        pixels.fill((0, 0, 255))
        pixels.show()
        await sleep(1)


async def theater_chase_rainbow():
    while True:
        for j in range(256):
            for q in range(3):
                for i in range(0, NUM_LEDS, 3):
                    pixels[i + q - 1] = wheel((i + j) % 255)

                pixels.show()
                await sleep(50 / 1000.0)

                for i in range(0, NUM_LEDS, 3):
                    pixels[i + q - 1] = (0, 0, 0)


async def fill_and_unfill(color):
    while True:
        for i in range(NUM_LEDS):
            pixels[i] = color
            pixels.show()
            await sleep(.001)
            
        for i in range(NUM_LEDS):
            pixels[i] = (0, 0, 0)
            pixels.show()
            await sleep(.001)
    

async def sparkle(color):
    while True:
        for i in range(NUM_LEDS):
            pixels[i] = color if i % 2 == 0 else (0, 0, 0)
        pixels.show()
        await sleep(.25)

        for i in range(NUM_LEDS):
            pixels[i] = color if i % 2 != 0 else (0, 0, 0)
        pixels.show()
        await sleep(.25)


async def rainbow_cycle():
    while True:
        for i in range(NUM_LEDS):
            pixels[i] = wheel(i % 255)
            pixels.show()
            await sleep(.001)

        for i in range(NUM_LEDS):
            pixels[i] = (0, 0, 0)
            pixels.show()
            await sleep(.001)


async def scrolling_text(color, text):
    iteration_count = -NUM_COLS
    image = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/roboto/Roboto-Regular.ttf", 14)

    text_width = draw.textsize(text, font)[0]

    while True:
        image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])
        draw.text((-iteration_count, 0), text, color, font = font)

        await create_task(display_image(image))

        iteration_count += 1

        if iteration_count > text_width:
            iteration_count = -NUM_COLS

        await sleep(.0001)


async def rainbow_scrolling_text(text):
    iteration_count = -NUM_COLS
    total_iterations = 0

    image = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/roboto/Roboto-Regular.ttf", 14)
    text_width = draw.textsize(text, font)[0]

    while True:
        image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])
        draw.text((-iteration_count, 0), text, wheel(total_iterations % 255), font = font)

        await create_task(display_image(image))

        iteration_count += 1
        if iteration_count > text_width:
            iteration_count = -NUM_COLS
            total_iterations += 10


async def load_image_url(url):
    image = Image.open(requests.get(url, stream = True).raw).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
    await display_image(image)


async def progress_pride_flag():
    await load_image_url("https://static.dezeen.com/uploads/2018/06/lgbt-pride-flag-redesign-hero.jpg")


async def camera():
    capture = cv2.VideoCapture(-1, cv2.CAP_V4L)
    success = capture.read()[0]

    try:
        while success:
            success, image = capture.read()

            if not success:
                return

            image = cv2.flip(image, 1)
            image = Image.fromarray(image).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
            
            r, g, b = image.split()
            image = Image.merge("RGB", (b, g, r))

            await create_task(display_image(image))
    finally:
        capture.release()


async def equalizer(color):

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format = AUDIO_FORMAT,
        channels = AUDIO_CHANNELS,
        rate = AUDIO_RATE,
        input_device_index = AUDIO_DEVICE_INDEX,
        input = True,
        frames_per_buffer = AUDIO_BLOCK_SIZE
    )

    try:
        while True:
            data = np.frombuffer(stream.read(AUDIO_BLOCK_SIZE, exception_on_overflow = False), dtype = np.int16)
            data = data - np.average(data)

            n = len(data)
            k = np.arange(n)

            tarr = n / float(AUDIO_RATE)

            frqarr = k / float(tarr)
            frqarr = frqarr[range(n // 2)]

            data = fft.fft(data) / n
            data = abs(data[range(n // 2)])

            y_values = []

            chunked_data = np.split(data, len(data) / (len(data) // NUM_COLS))

            for i in range(len(chunked_data)):
                y_value = np.average(chunked_data[i]).astype(np.int64)

                if i == len(chunked_data) - 1:
                    y_value //= 5

                y_values.append(y_value)

            for x in range(len(y_values)):
                value = y_values[x]

                for y in range(NUM_ROWS):
                    pixels[get_pixel_index_at(x, NUM_ROWS - y - 1)] = color if y <= value else (0, 0, 0)

            pixels.show()
            await sleep(.0001)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()



async def rainbow_equalizer():

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format = AUDIO_FORMAT,
        channels = AUDIO_CHANNELS,
        rate = AUDIO_RATE,
        input_device_index = AUDIO_DEVICE_INDEX,
        input = True,
        frames_per_buffer = AUDIO_BLOCK_SIZE
    )

    try:
        while True:
            data = np.frombuffer(stream.read(AUDIO_BLOCK_SIZE, exception_on_overflow = False), dtype = np.int16)
            data = data - np.average(data)

            n = len(data)
            k = np.arange(n)

            tarr = n / float(AUDIO_RATE)

            frqarr = k / float(tarr)
            frqarr = frqarr[range(n // 2)]

            data = fft.fft(data) / n
            data = abs(data[range(n // 2)])

            y_values = []

            chunked_data = np.split(data, len(data) / (len(data) // NUM_COLS))

            for i in range(len(chunked_data)):
                y_value = np.average(chunked_data[i]).astype(np.int64)

                if i == len(chunked_data) - 1:
                    y_value //= 5

                y_values.append(y_value)

            for x in range(len(y_values)):
                value = y_values[x]

                for y in range(NUM_ROWS):
                    pixels[get_pixel_index_at(x, NUM_ROWS - y - 1)] = wheel(((value - y) * 5) % 255) if y <= value else (0, 0, 0)

            pixels.show()
            await sleep(.0001)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
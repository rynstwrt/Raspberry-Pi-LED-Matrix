from VideoCapture import VideoCapture
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
import colorsys
from math import floor
import datetime
from os import getenv
from dotenv import load_dotenv
load_dotenv()
from random import randint


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


################### HELPER FUNCTIONS ###################
def hsv_wheel(index):
    index = index % 255
    r, g, b = colorsys.hsv_to_rgb(index / 255, 1, 1)
    return (floor(r * 255), floor(g * 255), floor(b * 255))


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


async def display_current_time_and_return_seconds(color):
    now = datetime.datetime.now()

    h = now.hour
    m = now.minute
    s = now.second

    if h > 12:
        h -= 12

    if h == 0:
        h = 12

    if h < 10:
        h = "0{}".format(h)

    if m < 10:
        m = "0{}".format(m)

    text = "{}:{}".format(h, m)

    image = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/roboto/Roboto-Regular.ttf", 12)
    text_width, text_height = draw.textsize(text, font)

    image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])
    draw.text((NUM_COLS / 2 - text_width / 2, NUM_ROWS / 2 - text_height / 2), text, color, font = font)

    await display_image(image)
    return s


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
                    pixels[i + q - 1] = hsv_wheel((i + j) % 255)

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
            pixels[i] = hsv_wheel(i)
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
        draw.text((-iteration_count, 0), text, hsv_wheel(total_iterations / 2), font = font)

        await create_task(display_image(image))

        iteration_count += 1
        total_iterations += 1
        if iteration_count > text_width:
            iteration_count = -NUM_COLS


async def load_image_url(url):
    image = Image.open(requests.get(url, stream = True).raw).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
    await display_image(image)


async def progress_pride_flag():
    await load_image_url("https://static.dezeen.com/uploads/2018/06/lgbt-pride-flag-redesign-hero.jpg")


async def camera():
    capture = VideoCapture()

    try:
        while True:
            image = capture.read()
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
                    pixels[get_pixel_index_at(x, NUM_ROWS - y - 1)] = hsv_wheel((value + y) / 2) if y <= value else (0, 0, 0)

            pixels.show()
            await sleep(.0001)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


async def clock(color):
    seconds = await display_current_time_and_return_seconds(color)
    await sleep(60 - seconds)

    while True:
        await display_current_time_and_return_seconds(color)
        await sleep(60)
    

async def get_joke(color):
    api_key = getenv("JOKE_API_KEY")

    headers = {
        "accept": "application/json"
    }

    resp = requests.get("https://dad-jokes.p.rapidapi.com/random/joke?rapidapi-key={}".format(api_key), headers = headers).json()
    
    setup = resp["body"][0]["setup"]
    punchline = resp["body"][0]["punchline"]
    joke = "{}          {}".format(setup, punchline)

    await scrolling_text(color, joke)


async def blocks():
    block_size = 4;

    while True:
        for x in range(0, NUM_COLS, block_size):
            for y in range(0, NUM_ROWS, block_size):

                block_color = (randint(0, 255), randint(0, 255), randint(0, 255))

                for i in range(block_size):
                    new_x = x + i

                    for j in range(block_size):
                        new_y = (NUM_ROWS - y - 1) - j
                        pixels[get_pixel_index_at(new_x, new_y)] = block_color

                pixels.show();
                await sleep(.2)


        for x in range(0, NUM_COLS, block_size):
            for y in range(0, NUM_ROWS, block_size):
                for i in range(block_size):
                    new_x = x + i

                    for j in range(block_size):
                        new_y = (NUM_ROWS - y - 1) - j
                        pixels[get_pixel_index_at(new_x, new_y)] = (0, 0, 0)

                pixels.show();
                await sleep(.2)


async def squiggle(color):
    image = Image.new("RGB", (NUM_COLS, NUM_ROWS), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    while True:
        image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])

        number_of_points = randint(2, 5)
        line_points = []
        for _ in range(number_of_points):
            line_points.append((randint(0, NUM_COLS), randint(0, NUM_ROWS)))

        draw.line(line_points, width = 2, fill = color, joint = "curve")
        await display_image(image)
        await sleep(.0001)


async def hypnotize(color):
    image = Image.new("RGB", (NUM_COLS, NUM_ROWS), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    num_steps = min(NUM_ROWS, NUM_COLS) // 2 + 1

    while True:
        for i in range(num_steps):
            image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])
            
            top_left = (i, i)
            bottom_right = (NUM_COLS - i - 1, NUM_ROWS - i - 1)

            draw.rectangle([top_left, bottom_right], outline = color)

            await display_image(image)

        pixels.fill((0, 0, 0))
        pixels.show()
        await sleep(.1)

        for i in range(num_steps - 1, 0, -1):
            image.paste((0, 0, 0), [0, 0, NUM_COLS, NUM_ROWS])
            
            top_left = (i, i)
            bottom_right = (NUM_COLS - i - 1, NUM_ROWS - i - 1)

            draw.rectangle([top_left, bottom_right], outline = color, width = 2)

            await display_image(image)

        pixels.fill((0, 0, 0))
        pixels.show()
        await sleep(.1)


async def snake(color):
    while True:
        for i in range(-6, NUM_LEDS + 6, 1):
            pixels.fill((0, 0, 0))

            for j in range(i - 5, i + 6, 1):
                distance_from_i = abs(i - j)

                if j > 0 and j < NUM_LEDS:
                    color_max_1 = list(map(lambda x: x // 255, color))
                    
                    color_hsv_1 = list(colorsys.rgb_to_hsv(*color_max_1))
                    color_hsv_1[2] = 1 - (distance_from_i / 5)

                    color_rgb_1 = colorsys.hsv_to_rgb(*color_hsv_1)
                    color_rgb = tuple(map(lambda x: floor(x * 255), color_rgb_1))

                    pixels[j] = color_rgb

            pixels.show()
            await sleep(.00001)


async def face_detection(color):
    capture = VideoCapture()

    try:
        while True:
            frame = capture.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            height, width, _ = frame.shape

            image = Image.new("RGB", (width, height), (0, 0, 0))
            opencv_image = np.array(image)[:, :, ::-1].copy()

            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            faces = face_classifier.detectMultiScale(gray, 1.5, 3)

            profile_classifier = cv2.CascadeClassifier("haarcascade_profileface.xml")
            profiles = profile_classifier.detectMultiScale(gray, 1.5, 3)

            for (x, y, w, h) in faces:
                cv2.rectangle(opencv_image, (x, y), (x + w, y + h), color, 2)

            if faces is ():
                for (x, y, w, h) in profiles:
                    cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image).resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)

            await create_task(display_image(image))
    finally:
        capture.release()


async def object_tracking():
    cap = VideoCapture()
    
    first_frame = cap.read()
    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255

    try:
        while True:
            frame = cap.read()
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(previous_gray, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            hsv[..., 0] = angle * (180 / (np.pi / 2))
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            final = Image.fromarray(final).resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)

            await create_task(display_image(final))
    finally:
        cap.release()
        

async def object_tracking_2():
    cap = VideoCapture()
    last_frame = None

    try:
        while True:
            original_frame = cap.read()
            original_frame = cv2.flip(original_frame, 1)
            
            if last_frame is None:
                last_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                continue

            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

            image_diff = cv2.absdiff(frame, last_frame)
            _, thresh = cv2.threshold(image_diff, 40, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            image = Image.fromarray(dilated).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
            await create_task(display_image(image))

            last_frame = frame
    finally:
        cap.release()
################### IMPORTS ###################
from BufferlessVideoCapture import BufferlessVideoCapture
import neopixel
from time import sleep, localtime, strftime
import cv2
from PIL import Image, ImageFont, ImageDraw
import requests
from math import ceil
import pyaudio
import numpy as np
np.set_printoptions(suppress=True)
from scipy import fft, arange



################### VARIABLES ###################
current_function = None
pixels = None
NUM_LEDS = None
COLOR_BLACK = (0, 0, 0)
NUM_ROWS = None
NUM_COLS = None



################### AUDIO VARIABLES ###################
AUDIO_DEVICE_INDEX = 0
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 2
AUDIO_RATE = 48000
AUDIO_BLOCK_SIZE = 1024 * 2



################### OBJECT DETECTION VARIABLES ###################
PATH_TO_CKPT = "/home/pi/Sample_TFLite_model/detect.tflite"
PATH_TO_LABELS = "/home/pi/Sample_TFLite_model/labelmap.txt"



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



################### MODE FUNCTIONS ###################
def dual_stripes(color1, color2):
    blank()
    while True:
        if current_function != None and current_function != dual_stripes:
            return

        for i in range(NUM_LEDS):
            if current_function != None and current_function != dual_stripes:
                return

            pixels[i] = color1
            pixels.show()

        for i in range(NUM_LEDS):
            if current_function != None and current_function != dual_stripes:
                return

            pixels[i] = color2
            pixels.show()

        

def rainbow_cycle():
    while True:
        if current_function != None and current_function != rainbow_cycle:
            return

        blank()

        for j in range(255):   
            for i in range(NUM_LEDS):
                if current_function != None and current_function != rainbow_cycle:
                    return

                pixel_index = (i * 256 // NUM_LEDS) + j
                pixels[i] = wheel(pixel_index & 255)
                pixels.show()
            sleep(2)



def theater_chase_rainbow(wait_ms=50):
    for j in range(256):
        for q in range(3):
            for i in range(0, NUM_LEDS, 3):
                pixels[i+q - 1] = wheel((i+j) % 255)

                if current_function != None and current_function != theater_chase_rainbow:
                    return

            pixels.show()
            sleep(wait_ms/1000.0)
            for i in range(0, NUM_LEDS, 3):
                if current_function != None and current_function != theater_chase_rainbow:
                    return
                pixels[i+q - 1] = (0, 0, 0)
    theater_chase_rainbow(wait_ms)



def load_image_url(url):
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)

    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            if current_function != None and current_function != load_image_url:
                return

            index = get_pixel_index_at(x, y)

            coords = x, y
            pixels[index] = img.getpixel(coords)
    
    pixels.show()



def load_image_camera(img_array, reverse = True):
    img = Image.fromarray(img_array).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))


    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            if current_function != None and current_function != load_camera:
                return

            index = get_pixel_index_at(x, y)

            coords = NUM_COLS - x - 1 if reverse else x, y
            pixels[index] = img.getpixel(coords)
    
    pixels.show()



def load_camera():
    video_capture = cv2.VideoCapture(-1, cv2.CAP_V4L)
    success, img = video_capture.read()

    while success:
        success, img = video_capture.read()

        if current_function != None and current_function != load_camera:
            return

        if not success:
            return

        load_image_camera(img)
    
    video_capture.release()



def text(passed_text, color):
    img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

    text_size = draw.textsize(passed_text, font=font)
    width = text_size[0]
    height = text_size[1]

    draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), passed_text, color, font = font)

    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            if current_function != None and current_function != text:
                return

            index = get_pixel_index_at(x, y)

            coords = x, y
            pixels[index] = img.getpixel(coords)
    
    pixels.show()



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
        if current_function != None and current_function != rainbow_text:
            return

        for x in range(NUM_COLS):
            for y in range(NUM_ROWS):
                pixel_index = get_pixel_index_at(x, y)
                coords = x, y

                color = COLOR_BLACK if img.getpixel(coords) == COLOR_BLACK else wheel(index + pixel_index & 255) 
                pixels[pixel_index] = color

        
        pixels.show()
        index += 2

        if index > 255:
            index = 0



def scrolling_text(text, color):
    index = -NUM_COLS
    global next_function

    while True:

        if current_function != None and current_function != scrolling_text:
            return

        img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 14)

        text_size = draw.textsize(text, font=font)
        width = text_size[0]
        height = text_size[1]

        draw.text((-index, ceil(NUM_ROWS / 2 - height / 2) - 1), text, color, font = font)

        for x in range(NUM_COLS):
            for y in range(NUM_ROWS):
                pixel_index = get_pixel_index_at(x, y)
                coords = x, y
                pixels[pixel_index] = img.getpixel(coords)
        
        pixels.show()
        index += 1

        if index > width:
            index = -NUM_COLS



def display_time(time_string, color):
    img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

    text_size = draw.textsize(time_string, font=font)
    width = text_size[0]
    height = text_size[1]

    draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), time_string, color, font = font)

    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            if current_function != None and current_function != clock:
                return

            index = get_pixel_index_at(x, y)

            coords = x, y
            pixels[index] = img.getpixel(coords)
    
    pixels.show()

    sleep(.02)



def get_time():
    local_time = localtime()
    h = int(strftime("%H", local_time))
    m = strftime("%M", local_time)
    s = strftime("%S", local_time)

    if h > 12:
        h = h - 12

    if h < 10 and len(str(h)) == 1:
        h = "0{}".format(h)

    return "{}:{}".format(h, m)



def clock(color):
    local_time = localtime()
    h = int(strftime("%H", local_time))
    m = strftime("%M", local_time)
    s = strftime("%S", local_time)

    if h > 12:
        h = h - 12
    
    if h < 10 and len(str(h)) == 1:
        h = "0{}".format(h)
    
    display_time("{}:{}".format(h, m), color)
    sleep(60 - int(s))

    if current_function != None and current_function != clock:
        return

    display_time(get_time(), color)

    while True:
        sleep(60)
        if current_function != None and current_function != clock:
            return
        display_time(get_time(), color)



def sparkle(color):
    while True:
        for i in range(NUM_LEDS):
            if current_function != None and current_function != sparkle:
                break
            
            pixel_color = color if i % 2 == 0 else COLOR_BLACK
            pixels[i] = pixel_color

        pixels.show()
        sleep(.2)

        for i in range(NUM_LEDS):
            if current_function != None and current_function != sparkle:
                break

            pixel_color = color if i % 2 != 0 else COLOR_BLACK
            pixels[i] = pixel_color

        pixels.show()
        sleep(.2)



def temperature(city, state, country, api_key, units):
    while True:
        if current_function != None and current_function != temperature:
            return

        response = requests.get("http://api.openweathermap.org/data/2.5/weather?q={},{},{}&appid={}&units={}".format(city, state, country, api_key, units))
        if (response.status_code != 200):
            pixels.fill((255, 0, 0))
            pixels.show()
            print(response.json())
            return

        temp = response.json()["main"]["temp"]
        temperature_text = "{:.1f}°".format(temp)

        color = None
        if temp > 90:
            color = (255, 0, 0)
        elif temp < 55:
            color = (31, 154, 255)
        else:
            color = (255, 255, 255)

        img = Image.new("RGB", (NUM_COLS, NUM_ROWS), 0)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

        text_size = draw.textsize(temperature_text, font=font)
        width = text_size[0]
        height = text_size[1]

        draw.text((ceil(NUM_COLS / 2 - width / 2), ceil(NUM_ROWS / 2 - height / 2) - 1), temperature_text, color, font = font)

        for x in range(NUM_COLS):
            for y in range(NUM_ROWS):
                if current_function != None and current_function != temperature:
                    return

                index = get_pixel_index_at(x, y)

                coords = x, y
                pixels[index] = img.getpixel(coords)
        
        pixels.show()
        sleep(320) # 5 minutes



def load_image_video(img_array):
    img = Image.fromarray(img_array).convert("RGB").resize((NUM_COLS, NUM_ROWS), Image.ANTIALIAS)
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))


    for x in range(NUM_COLS):
        for y in range(NUM_ROWS):
            if current_function != None and current_function != load_video_url:
                return

            index = get_pixel_index_at(x, y)

            coords = x, y
            pixels[index] = img.getpixel(coords)
    
    pixels.show()



def load_video_url(url):
    video_capture = cv2.VideoCapture(url)
    success, img = video_capture.read()

    while success:
        success, img = video_capture.read()

        if current_function != None and current_function != load_video_url:
            return

        if not success:
            return
        
        load_image_video(img)



def equalizer(color):
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format = AUDIO_FORMAT, 
        channels = AUDIO_CHANNELS,
        rate = AUDIO_RATE,
        input_device_index = AUDIO_DEVICE_INDEX,
        input = True,
        frames_per_buffer = AUDIO_BLOCK_SIZE)

    while True:
        if current_function != None and current_function != equalizer:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return

        data = np.fromstring(stream.read(AUDIO_BLOCK_SIZE, exception_on_overflow = False), dtype = np.int16)

        data = data - np.average(data)

        n = len(data)
        k = arange(n)

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
                pixels[get_pixel_index_at(x, NUM_ROWS - y - 1)] = color if y <= value else COLOR_BLACK
        
        pixels.show()



def rainbow_equalizer():
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format = AUDIO_FORMAT, 
        channels = AUDIO_CHANNELS,
        rate = AUDIO_RATE,
        input_device_index = AUDIO_DEVICE_INDEX,
        input = True,
        frames_per_buffer = AUDIO_BLOCK_SIZE)

    while True:
        if current_function != None and current_function != rainbow_equalizer:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            return

        data = np.fromstring(stream.read(AUDIO_BLOCK_SIZE, exception_on_overflow = False), dtype = np.int16)

        data = data - np.average(data)

        n = len(data)
        k = arange(n)

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
                pixels[get_pixel_index_at(x, NUM_ROWS - y - 1)] = wheel(((x + y) * (255 // NUM_ROWS)) & 255) if y <= value else COLOR_BLACK
        
        pixels.show()

    

has_imported_obj_detection = False
def object_detection(color, thresh =  0.45):
    
    global has_imported_obj_detection

    if not has_imported_obj_detection:
        has_imported_obj_detection = True
        from tflite_runtime.interpreter import Interpreter

    labels = []

    with open(PATH_TO_LABELS, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == "???":
        del(labels[0])

    interpreter = Interpreter(model_path = PATH_TO_CKPT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    cap = BufferlessVideoCapture(-1)

    while True:
        frame1 = cap.read()

        if current_function != None and current_function != object_detection:
            cap.release()
            return

        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis = 0)

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        classes = interpreter.get_tensor(output_details[1]["index"])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence 

        highest_score = 0
        current_label = None

        for i in range(len(scores)):
            if ((scores[i] > thresh) and (scores[i] <= 1.0)):
                if highest_score < scores[i]:
                    highest_score = scores[i]
                    current_label = labels[int(classes[i])]

        if current_label == None:
            continue

        img = Image.new("RGB", (NUM_COLS, NUM_ROWS))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)
        text_width = draw.textsize(current_label, font = font)[0]

        x_pos = text_width + NUM_COLS
        while x_pos > -text_width - NUM_COLS:
            img = Image.new("RGB", (NUM_COLS, NUM_ROWS))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/user/share/fonts/truetype/noto/NotoMono-Regular.ttf", 10)

            text_size = draw.textsize(current_label, font=font)
            text_width = text_size[0]
            text_height = text_size[1]

            draw.text((x_pos, ceil(NUM_ROWS / 2 - text_height / 2) - 1), current_label, color, font = font)

            for x in range(NUM_COLS):
                for y in range(NUM_ROWS):
                    if current_function != None and current_function != object_detection:
                        return

                    index = get_pixel_index_at(x, y)

                    coords = x, y
                    pixels[index] = img.getpixel(coords)
            
            x_pos -= 1

            pixels.show()

    

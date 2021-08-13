from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import modes
from pydantic import BaseModel
from typing import Optional
import uvicorn
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"]
)


@app.get("/getmodes")
async def on_get_modes():
    return json.loads(modes.MODES)


modes_dict = {
    "off": modes.off,
    "rgb": modes.rgb,
    "theater chase rainbow": modes.theater_chase_rainbow,
    "fill and unfill": modes.fill_and_unfill,
    "sparkle": modes.sparkle,
    "rainbow cycle": modes.rainbow_cycle,
    "scrolling text": modes.scrolling_text,
    "rainbow scrolling text": modes.rainbow_scrolling_text,
    "load image url": modes.load_image_url,
    "progress pride flag": modes.progress_pride_flag,
    "camera": modes.camera,
    "equalizer": modes.equalizer,
    "rainbow equalizer": modes.rainbow_equalizer
}


modes_arg_type_dict = {
    "fill and unfill": ["color"],
    "sparkle": ["color"],
    "scrolling text": ["color", "text"],
    "rainbow scrolling text": ["text"],
    "load image url": ["text"],
    "equalizer": ["color"]
}


def get_arguments_for_mode(name, color, text):
    if not name in modes_arg_type_dict:
        return False

    arguments = []

    types = modes_arg_type_dict[name]
    for type in types:
        if type == "color":
            arguments.append(color)
        elif type == "text":
            arguments.append(text)

    return arguments


class ModeFromPost(BaseModel):
    mode_name: str
    mode_color: Optional[str]
    mode_text: Optional[str]


current_task = None
# loop = asyncio.get_event_loop() 
@app.post("/setmode")
async def on_set_mode(mode_from_post: ModeFromPost):
    global current_task

    # reset for the new pattern
    if current_task is not None:
        current_task.cancel()
    
    for i in range(len(modes.pixels)):
        modes.pixels[i] = (0, 0, 0)

    # get the POSTed pattern's data
    name = mode_from_post.mode_name
    color = tuple(int(mode_from_post.mode_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    text = mode_from_post.mode_text

    # start the pattern
    arguments = get_arguments_for_mode(name, color, text)
    current_task = asyncio.create_task(modes_dict[name](*arguments)) if arguments else asyncio.create_task(modes_dict[name]())


if __name__ == "__main__":
    uvicorn.run("led_matrix:app", host = "10.164.1.125", port = 8000, reload = True)
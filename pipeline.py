import threading
import pyaudio
import timing
import struct
import array
import torch
import time
import math
import sys
import os

RATE = 22050
BLOCK = 1024
IN_BUFF = 16
CHANNELS = 1

current_milli_time = lambda: int(round(time.time() * 1000))

raw_audio = None
output = None

def format_read(read):
    read = torch.tensor(struct.unpack(int(len(read)/4)*'f', read))
    return read

def format_write(write):
    write = struct.pack(int(write.size(0))*'f', *write.tolist())
    return write

p = pyaudio.PyAudio()

thread_running = True;

def start_reading():
    global raw_audio
    while thread_running:
        before_time = current_milli_time()
        raw_audio = r_stream.read(BLOCK)
        print(current_milli_time() - before_time)

def start_writing():
    global output
    while output is None:
        pass
    while thread_running:
        output_copy = output
        w_stream.write(output_copy)

r_stream = p.open(
    format = pyaudio.paFloat32,
    channels = CHANNELS,
    rate = RATE,
    input = True,
)

w_stream = p.open(
    format = pyaudio.paFloat32,
    channels = CHANNELS,
    rate = RATE,
    output = True,
)

r_stream.start_stream()
w_stream.start_stream()

print("Loading model...")
with torch.no_grad():
    model = timing.DenoiseNetwork().to(timing.device).eval()
    model.load_state_dict(torch.load("model.pt"))
    timing.run_model(torch.zeros(1, 1, IN_BUFF*BLOCK).to(timing.device), model)

    tensor_buffer = torch.zeros(IN_BUFF*BLOCK).float().to(timing.device)

    before_time = 0
    after_time = 0

    read_thread = threading.Thread(target = start_reading, daemon = True)
    read_thread.start()

    write_thread = threading.Thread(target = start_writing, daemon = True)
    write_thread.start()

    print("Model loaded.")

    raw_audio_copy = raw_audio
    while True:
        if not raw_audio_copy == raw_audio:
            before_time = current_milli_time()
            raw_audio_copy = raw_audio

            tensor_sound = format_read(raw_audio_copy).to(timing.device)[:IN_BUFF*BLOCK]
            tensor_buffer = torch.cat((tensor_buffer[BLOCK:], tensor_sound))

            processed = timing.run_model(tensor_buffer.view(1, 1, -1), model).cpu().view(-1)

            output = format_write(processed)
            after_time = current_milli_time()
            #print(after_time - before_time)

    thread_running = False;
    read_thread.join()
    write_thread.join()

    r_stream.stop_stream()
    r_stream.close()
    w_stream.stop_stream()
    w_stream.close()
    p.terminate()

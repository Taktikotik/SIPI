import json
import soundfile
import wave
import yandex_tracker_client as ya
from revChatGPT.V1 import Chatbot
from pathlib import Path
from aiogram import Bot, Dispatcher, executor, types
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from PIL import Image

GPT_key = ('ChatGPT_KEY')  # Перейдите по ссылке
# https://chat.openai.com/api/auth/session и скопируйте значение "accessToken"

BOT_API = 'Bot_API'  # Впишите API телеграмм бота

model = Model('vosk-model-small-ru-0.22')  # Имя папки vosk модели
rec = KaldiRecognizer(model, 44100)

path = "/files/voices"  # Путь до места хранение временного файла голосового сообщения

GPT_use = False
GPT_ask = False
punctuation = False

bot = Bot(token=BOT_API)
dp = Dispatcher(bot)

#  Клавиатуры
kb = types.ReplyKeyboardMarkup([[types.KeyboardButton('📝 Создать новую задачу.')]], resize_keyboard=True)

followers_arr = []

names = {
    0: '-',
    1: '.',
    2: '0',
    3: '1',
    4: '2',
    5: '3',
    6: '4',
    7: '5',
    8: '6',
    9: '7',
    10: '8',
    11: '9',
    12: 'num'
}

model_1 = YOLO("number_m_26p_6500e.pt")  # Модель для распознавания значения модуля
model_2 = YOLO('digit_wo_m_40p_200e.pt')  # Модель для распознавания цифр

# Реакция бота на команду /start
@dp.message_handler(commands=['start'])
async def send_hello(message: types.Message):
    await message.reply('Привет! Зачем ты здесь? Я могу превратить русскую речь из твоего голосового сообщения в '
                        'текст и ответить на него. Также я могу распознавать числа с картинок. Напите "/help" '
                        'для получения информации о настройках.', reply_markup=kb)


# Реакция бота на команду /help
@dp.message_handler(commands=['help'])
async def send_help(message: types.Message):
    await message.reply('Нужна помощь? Напомню, что я могу превратить русскую речь из твоего голосового сообщения в '
                        'текст и ответить на него. Также я могу распознавать числа с картинок.\n')
    #Убранно в связи с не работой chatGPT
    """
                        'Напишите "/gpt_use" чтобы выключить/включить все функции Chat_gpt.\n'
                        'Напишите "/gpt_ask" чтобы выключить/включить задачу вопроса Chat_gpt.\n'
                        'Напишите "/punctuation" чтобы выключить/включить исправление пунктуации.')
    """

"""
@dp.message_handler(commands=['gpt_use'])
async def main_gpt(message: types.Message):
    global GPT_use, punctuation, GPT_ask
    GPT_use = not GPT_use
    punctuation = GPT_use
    GPT_ask = GPT_use
    if GPT_use:
        await message.reply('ChatGPT включён')
    else:
        await message.reply('ChatGPT выключён')


@dp.message_handler(commands=['gpt_ask'])
async def gpt_asking_status(message: types.Message):
    global GPT_ask
    GPT_ask = not GPT_ask
    if GPT_ask:
        await message.reply('Задача вопроса Chat_gpt включёна')
    else:
        await message.reply('Задача вопроса Chat_gpt выключёна')


@dp.message_handler(commands=['punctuation'])
async def punctuation_status(message: types.Message):
    global punctuation
    punctuation = not punctuation
    if punctuation:
        await message.reply('Проверка пунктуации включена')
    else:
        await message.reply('Проверка пунктуации выключена')
"""

# Реакция бота на текстовое сообщение
@dp.message_handler(content_types=types.ContentType.TEXT)
async def make_task(message: types.Message):
    await message.answer(message.text)


# Реакция бота на голосовое сообщение
@dp.message_handler(content_types=types.ContentType.VOICE)
async def answer_to_voice(message: types.Message):
    voice = await message.voice.get_file()
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    await bot.download_file(file_path=voice.file_path, destination=f'{path}/{voice.file_id}.wav')
    await bot.download_file(file_path=voice.file_path, destination=f'{path}/qwerty.wav')
    data, samplerate = soundfile.read(f'{path}/qwerty.wav')
    soundfile.write(f'{path}/qwerty.wav', data, samplerate)
    wf = wave.open(rf'{path}/qwerty.wav')
    result = ''
    last_n = False
    while True:
        data = wf.readframes(44100)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if res['text'] != '':
                result += f" {res['text']}"
                last_n = False
            elif not last_n:
                result += '\n'
                last_n = True
    res = json.loads(rec.FinalResult())
    result += f" {res['text']}"
    await message.delete()
    if GPT_use:
        if punctuation:
            await message.answer_chat_action("typing")
            chatbot = Chatbot(config={"access_token": GPT_key})
            *_, result = chatbot.ask(f'исправь пунктуацию: {result}')
            await message.answer(result['message'])
        else:
            await message.answer(result)
        if GPT_ask:
            await message.answer_chat_action("typing")
            chatbot = Chatbot(config={"access_token": GPT_key})
            *_, last = chatbot.ask(result['message'])
            await message.answer(last['message'])
    else:
        await message.answer(result)

# Реакция бота на фотографию
@dp.message_handler(content_types=types.ContentType.PHOTO)
async def answer_to_photo(message: types.Message):
    file = await bot.download_file_by_id(file_id=message.photo[-1].file_id)
    im = Image.open(file)  # Фотография
    # Кроп значения
    results = model_1.predict(im, stream=True, save=True)
    for result in results:
        coord = str(result.boxes.xyxy[0])
        coord = coord[coord.find('[') + 1:coord.find(']')]
        coord = coord.split(', ')
        for i in range(4):
            coord[i] = float(coord[i])
        coord = tuple(coord)
        crop = im.crop(coord)
    # Распознавание числа
    results = model_2.predict(crop, stream=True)
    for result in results:
        coord = str(result.boxes.xyxy)
        coord = coord[coord.find('[[') + 2:coord.find(']]')]
        coord = coord.split('],')
        for i in range(len(coord)):
            if '\n        [' in coord[i]:
                coord[i] = coord[i].replace('\n        [', '')
            coord[i] = coord[i].split(', ')
            for j in range(4):
                coord[i][j] = float(coord[i][j])
        classes = str(result.boxes.cls)
        classes = classes[classes.find('[') + 1:classes.find(']')]
        while ' ' in classes or '.' in classes:
            classes = classes.replace(' ', '')
            classes = classes.replace('.', '')
        classes = classes.split(',')
        for i in range(len(classes)):
            coord[i].append(names[int(classes[i])])
        coord.sort()
        num = ''
        for i in range(len(coord)):
            if coord[i][4] != 'num':
                num += coord[i][4]
            if coord[i][4] == '-' and i > 0:
                continue
        num = float(num)
        await message.answer(num)


# Реакция бота на непредусмотренные сообщения
@dp.message_handler(content_types=[types.ContentType.ANY])
async def answer_to_voice(message: types.Message):
    await message.answer('Я ещё не умею работать с такими данными.')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

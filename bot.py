import io
import telebot
from telebot import types
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from model_class import Model

TOKEN = 'тут был токен'
bot = telebot.TeleBot(TOKEN)

torch.manual_seed(42)
model_name = "IlyaGusev/saiga_llama3_8b"
DEFAULT_SYSTEM_PROMPT = """тут был длинный промпт"""

model = Model(model_name, DEFAULT_SYSTEM_PROMPT)


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    bot.send_message(message.chat.id, f'{message.from_user.first_name}, отправьте сюда текст для суммаризации.', reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text(message):
    text = message.text
    bot.reply_to(message, 'Спасибо! Обрабатываю ваш текст...')

    summary = model(text)
    
    bot.send_message(message.chat.id, summary)

bot.polling(non_stop=True)

from aiogram import F, Router, types, flags
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from aiogram.fsm.context import FSMContext
from aiogram.types.callback_query import CallbackQuery
from aiogram import Bot
import threading

import asyncio
import config

import kb
import text
import utils
from states import Gen

router = Router()


users_buffer = {}

async def send_photo(photo):
    bot = Bot(token=config.BOT_TOKEN)



class InfoAboutUser:
    def __init__(self):
        self.settings = {'num_epochs': 50, 'imsize': 256}
        self.photos = []
        self.type_algo = None

    def restart(self, algo=None):
        self.type_algo = algo

    def get_type_algo(self):
        return self.type_algo
    

@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet.format(name=msg.from_user.full_name), reply_markup=kb.menu)

@router.message(F.text == "Menu")
@router.message(F.text == "Go to menu")
@router.message(F.text == "◀️ Go to menu")
async def menu(msg: Message):
    await msg.answer(text.menu, reply_markup=kb.menu)



@router.callback_query(F.data == "art_st")
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.context_photo)
    await clbck.message.edit_text(text.content_photo)


    if clbck.from_user.id not in users_buffer:
        users_buffer[clbck.from_user.id] = InfoAboutUser()

    users_buffer[clbck.from_user.id].restart("art_st")

@router.callback_query(F.data == "photorealistic_st")
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.context_photo)
    await clbck.message.edit_text(text.content_photo)


    if clbck.from_user.id not in users_buffer:
        users_buffer[clbck.from_user.id] = InfoAboutUser()

    users_buffer[clbck.from_user.id].restart("photorealistic_st")
        


        

@router.message(Gen.context_photo)
@flags.chat_action("load_content_photo")
async def get_content_photo(msg: Message, state: FSMContext,  bot: Bot):
    
    
    file = await bot.get_file(msg.photo[-1].file_id)
    file_path = file.file_path
    photo = await bot.download_file(file_path)
    users_buffer[msg.chat.id].photos.append(photo)

    await state.set_state(Gen.style_photo)
    await msg.answer(text.style_photo)    




@router.message(Gen.style_photo)
@flags.chat_action("load")
async def generate_image(msg: Message, state: FSMContext, bot: Bot):
    
    
    file = await bot.get_file(msg.photo[-1].file_id)
    file_path = file.file_path
    photo = await bot.download_file(file_path)
    users_buffer[msg.chat.id].photos.append(photo)



    mesg = await msg.answer(text.gen_wait)
    if users_buffer[msg.chat.id].get_type_algo() == "photorealistic_st":
        utils.make_photorealistic_style_transfer(*users_buffer[msg.chat.id].photos)
    elif users_buffer[msg.chat.id].get_type_algo() == "art_st":
        utils.make_art_style_transfer(*users_buffer[msg.chat.id].photos)
    await mesg.delete()

    with open('result.jpg', 'rb') as image_from_buffer:
        await msg.answer_photo(BufferedInputFile(image_from_buffer.read(), filename='image_from_buffer.jpg*'))
            

    users_buffer[msg.chat.id].photos = []
    await msg.answer(text.after_st, reply_markup=kb.menu)
    

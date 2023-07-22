from aiogram import F, Router, types, flags
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.types.callback_query import CallbackQuery

import kb
import text
import utils
from states import Gen

router = Router()

@router.message(Command("start"))
async def start_handler(msg: Message):
    await msg.answer(text.greet.format(name=msg.from_user.full_name), reply_markup=kb.menu)

@router.message(F.text == "Menu")
@router.message(F.text == "Go to menu")
@router.message(F.text == "◀️ Go to menu")
async def menu(msg: Message):
    await msg.answer(text.menu, reply_markup=kb.menu)



@router.callback_query(F.data == "st")
async def input_text_prompt(clbck: CallbackQuery, state: FSMContext):
    await state.set_state(Gen.context_photo)
    await clbck.message.edit_text(text.content_photo)

@router.message(Gen.context_photo)
@flags.chat_action("load_content_photo")
async def generate_text(msg: Message, state: FSMContext):
    content_photo = msg.photo
    
    await utils.load_content_photo(content_photo)
    await msg.answer(text.style_photo)
    
    await state.set_state(Gen.style_photo)
    


@router.message(Gen.style_photo)
@flags.chat_action("load")
async def generate_image(msg: Message, state: FSMContext):
    style_photo = msg.photo
    mesg = await msg.answer(text.gen_wait)
    res = await utils.make_style_transfer(style_photo)
  
   
    await mesg.edit_text(res)
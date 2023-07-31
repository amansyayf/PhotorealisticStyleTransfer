from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
menu = [
    [InlineKeyboardButton(text="Photorealistic style transfer", callback_data="photorealistic_st"),
     InlineKeyboardButton(text="Art style transfer", callback_data="art_st")]
]
menu = InlineKeyboardMarkup(inline_keyboard=menu)
exit_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="◀️ Go to menu")]], resize_keyboard=True)
iexit_kb = InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="◀️ Go to menu", callback_data="menu")]])

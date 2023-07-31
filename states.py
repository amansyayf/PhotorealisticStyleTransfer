from aiogram.fsm.state import StatesGroup, State


class Gen(StatesGroup):
    context_photo = State()
    style_photo = State()
